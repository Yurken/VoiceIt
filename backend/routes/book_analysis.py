# ruff: noqa: RUF001
"""Book analysis endpoint (B4).

Provides POST /books/{book_id}/analyze — enqueues the GPU-coordinated
literary-analysis + voice-casting background task and returns 202 immediately.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import database, models
from ..database import get_db
from ..services import book_analysis

router = APIRouter(prefix="/books", tags=["books"])


@router.post(
    "/{book_id}/analyze",
    status_code=202,
    response_model=models.AnalyzeResponse,
)
async def analyze_book(
    book_id: str,
    body: models.AnalyzeRequest = models.AnalyzeRequest(),
    db: Session = Depends(get_db),
) -> models.AnalyzeResponse:
    """Enqueue literary analysis + voice casting for a book.

    - Returns **202** immediately with ``status="analyzing"`` and a ``task_id``.
    - Returns **404** if the book does not exist.
    - Returns **409** if the book is already being analyzed or generating audio.
    """
    book = db.query(database.Book).filter_by(id=book_id).first()
    if book is None:
        raise HTTPException(status_code=404, detail="未找到这本书")

    if book.status in ("analyzing", "generating"):
        raise HTTPException(
            status_code=409,
            detail="这本书正在分析，请等待完成" if book.status == "analyzing" else "这本书正在生成音频，请等待完成",
        )

    task_id = book_analysis.enqueue_analysis(
        book_id,
        body.model_size,
        body.narrator_voice_id,
        voice_mode=body.voice_mode if "voice_mode" in body.model_fields_set else None,
    )

    return models.AnalyzeResponse(
        book_id=book_id,
        task_id=task_id,
        status="analyzing",
    )
