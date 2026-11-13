"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import traceback
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Literal

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()


@dataclass
class GenerationJob:
    """Queued generation work plus the generation ID it belongs to."""

    generation_id: str
    coro: Coroutine
    group_id: str = "manual"
    concurrency: int = 1


# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()
_generation_worker_tasks: list[asyncio.Task] = []
_group_semaphores: dict[str, asyncio.Semaphore] = {}
_queue_loop: asyncio.AbstractEventLoop | None = None


class _OwnedCoroutine(Coroutine):
    """Close a completion hook and the unstarted coroutine it owns together."""

    def __init__(self, coro, inner):
        self.coro = coro
        self.inner = inner

    def __await__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        return self.coro.send(value)

    def throw(self, typ, val=None, tb=None):
        return self.coro.throw(typ, val, tb)

    def close(self):
        try:
            self.coro.close()
        finally:
            self.inner.close()


def with_coro_cleanup(coro, inner):
    return _OwnedCoroutine(coro, inner)


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _generation_worker(queue=None, running=None, queued=None, cancelled=None, semaphores=None):
    """Worker that processes generation tasks one at a time."""
    queue = _generation_queue if queue is None else queue
    running = _running_generation_tasks if running is None else running
    queued = _queued_generation_ids if queued is None else queued
    cancelled = _cancelled_generation_ids if cancelled is None else cancelled
    semaphores = _group_semaphores if semaphores is None else semaphores
    while True:
        job = await queue.get()
        try:
            if job.generation_id in cancelled:
                cancelled.discard(job.generation_id)
                job.coro.close()
                continue

            semaphore_key = f"{job.group_id}:{job.concurrency}"
            semaphore = semaphores.setdefault(semaphore_key, asyncio.Semaphore(job.concurrency))

            async def run_job(current_job=job, current_semaphore=semaphore):
                try:
                    async with current_semaphore:
                        await current_job.coro
                finally:
                    # Also close a coroutine cancelled before it entered the semaphore.
                    current_job.coro.close()

            task = asyncio.create_task(run_job())
            running[job.generation_id] = task
            queued.discard(job.generation_id)
            try:
                await task
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
        except Exception:
            traceback.print_exc()
            await _force_fail_if_active(
                job.generation_id,
                "Worker exited without writing terminal status",
            )
        finally:
            job.coro.close()
            running.pop(job.generation_id, None)
            queued.discard(job.generation_id)
            queue.task_done()


async def _force_fail_if_active(generation_id: str, error: str) -> None:
    """Best-effort recovery — flip an active row to failed if the worker
    bailed before writing a terminal status. Catches the case where the gen
    coroutine's own status-write raised (e.g. SQLite lock contention)."""
    try:
        from ..database import Generation as DBGeneration, get_db
        from . import history

        db = next(get_db())
        try:
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return
            if (gen.status or "completed") not in ("pending", "loading_model", "generating"):
                return
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=db,
                error=error,
            )
            from .book_generation import settle_cancelled_book_generation

            settle_cancelled_book_generation(generation_id, db)
        finally:
            db.close()
    except Exception:
        traceback.print_exc()


def enqueue_generation(generation_id: str, coro, *, group_id: str = "manual", concurrency: int = 1):
    """Add a generation coroutine to the serial queue."""
    if _generation_queue is None:
        raise RuntimeError("Generation queue has not been initialized")

    queue = _generation_queue
    queued_ids = _queued_generation_ids
    queued_ids.add(generation_id)
    job = GenerationJob(generation_id=generation_id, coro=coro, group_id=group_id, concurrency=max(1, min(concurrency, 4)))
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    def accept_job():
        if queue is not _generation_queue:
            job.coro.close()
            queued_ids.discard(generation_id)
            create_background_task(_force_fail_if_active(generation_id, "Generation queue restarted"))
            return
        queue.put_nowait(job)

    if _queue_loop is not None and current_loop is not _queue_loop:
        _queue_loop.call_soon_threadsafe(accept_job)
    else:
        accept_job()


def cancel_generation(generation_id: str) -> Literal["queued", "running"] | None:
    """Cancel a queued or running generation if it is still active."""
    running_task = _running_generation_tasks.get(generation_id)
    if running_task is not None:
        running_task.cancel()
        return "running"

    if generation_id in _queued_generation_ids:
        _queued_generation_ids.discard(generation_id)
        _cancelled_generation_ids.add(generation_id)
        return "queued"

    return None


def init_queue(force: bool = False):
    """Initialize the generation queue and start the worker.

    Must be called once during application startup (inside a running event loop).
    """
    global _generation_queue, _generation_worker_task
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids
    global _generation_worker_tasks, _group_semaphores, _queue_loop

    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not force:
            return
        for worker in _generation_worker_tasks:
            worker.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()
    if force and _generation_queue is not None:
        while True:
            try:
                job = _generation_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            job.coro.close()
            _generation_queue.task_done()
            create_background_task(_force_fail_if_active(job.generation_id, "Generation queue restarted"))

    _generation_queue = asyncio.Queue()
    _queue_loop = asyncio.get_running_loop()
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _group_semaphores = {}
    _generation_worker_tasks = [create_background_task(_generation_worker(_generation_queue, _running_generation_tasks, _queued_generation_ids, _cancelled_generation_ids, _group_semaphores)) for _ in range(4)]
    _generation_worker_task = _generation_worker_tasks[0]
