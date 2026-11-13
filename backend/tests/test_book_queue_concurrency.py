import asyncio

import pytest

from backend.services import task_queue


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2, 4])
async def test_book_queue_runs_with_the_requested_bound(concurrency):
    task_queue.init_queue(force=True)
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0
    completed = []

    async def job(number):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == concurrency:
            started.set()
        try:
            await release.wait()
            completed.append(number)
        finally:
            active -= 1

    try:
        for number in range(8):
            task_queue.enqueue_generation(str(number), job(number), group_id="book", concurrency=concurrency)
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        assert peak == concurrency
        release.set()
        await asyncio.wait_for(task_queue._generation_queue.join(), 2)
        assert sorted(completed) == list(range(8))
        assert peak == concurrency
    finally:
        for worker in task_queue._generation_worker_tasks:
            worker.cancel()
        await asyncio.gather(*task_queue._generation_worker_tasks, return_exceptions=True)
