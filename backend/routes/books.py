# ruff: noqa: RUF001
"""Book CRUD and import endpoints (contract 01).

Wires: POST /books/import, GET /books, GET /books/{id},
       PATCH /books/{id}, DELETE /books/{id}.

POST /books/{id}/analyze lives in routes/book_analysis.py (B4).
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import database, models
from ..config import get_data_dir, to_storage_path
from ..database import get_db
from ..services import ingestion
from ..services.book_overview import chapter_generation_state

router = APIRouter(prefix="/books", tags=["books"])

ALLOWED = {"epub", "fb2", "txt", "pdf"}
MAX_BYTES = 200 * 1024 * 1024  # 200 MB


# ---------------------------------------------------------------------------
# Internal helpers (inline per task spec; B4 moves persistence to books.py)
# ---------------------------------------------------------------------------


def _get_book_or_404(book_id: str, db: Session) -> database.Book:
    """Return a Book row or raise 404."""
    book = db.query(database.Book).filter_by(id=book_id).first()
    if book is None:
        raise HTTPException(status_code=404, detail="未找到这本书")
    return book


def _chapter_count(book_id: str, db: Session) -> int:
    return db.query(database.Chapter).filter_by(book_id=book_id).count()


def _chapter_to_summary(chapter: database.Chapter, db: Session) -> models.ChapterSummary:
    return models.ChapterSummary(
        id=chapter.id,
        number=chapter.number,
        title=chapter.title,
        word_count=chapter.word_count,
        story_id=chapter.story_id,
        generation_state=chapter_generation_state(chapter.id, db),
    )


def _book_to_response(book: database.Book, db: Session) -> models.BookResponse:
    return models.BookResponse(
        id=book.id,
        title=book.title,
        author=book.author,
        source_format=book.source_format,
        cover_path=book.cover_path,
        status=book.status,
        chapter_count=_chapter_count(book.id, db),
        created_at=book.created_at,
        updated_at=book.updated_at,
        **_export_fields(book.id),
    )


def _book_to_detail(book: database.Book, db: Session) -> models.BookDetailResponse:
    chapters = db.query(database.Chapter).filter_by(book_id=book.id).order_by(database.Chapter.number).all()
    chapter_summaries = [_chapter_to_summary(ch, db) for ch in chapters]
    return models.BookDetailResponse(
        id=book.id,
        title=book.title,
        author=book.author,
        source_format=book.source_format,
        cover_path=book.cover_path,
        status=book.status,
        chapter_count=len(chapters),
        created_at=book.created_at,
        updated_at=book.updated_at,
        chapters=chapter_summaries,
        **_export_fields(book.id),
    )


def _export_fields(book_id: str) -> dict:
    from ..services.book_export_api import get_export_entry

    entry = get_export_entry(book_id)
    if not entry:
        return {}
    formats = {".m4b": "m4b", ".mp3": "mp3_single", ".zip": "mp3_per_chapter"}
    extension = Path(entry["filename"]).suffix.lower()
    return {"export_filename": entry["filename"], "export_format": formats.get(extension)}


def _persist_parsed_book(
    parsed: ingestion.ParsedBook,
    db: Session,
) -> database.Book:
    """Persist a ParsedBook (book + chapters + cover) and return the ORM Book."""
    # Persist cover bytes if present
    cover_path: str | None = None
    if parsed.cover_bytes:
        data_dir = get_data_dir()
        covers_dir = data_dir / "covers"
        covers_dir.mkdir(parents=True, exist_ok=True)
        cover_file = covers_dir / f"{uuid.uuid4()}.jpg"
        cover_file.write_bytes(parsed.cover_bytes)
        cover_path = to_storage_path(cover_file)

    book = database.Book(
        title=parsed.title,
        author=parsed.author,
        source_format=parsed.source_format,
        cover_path=cover_path,
        status="imported",
    )
    db.add(book)
    db.flush()  # populate book.id before child inserts

    for ch in parsed.chapters:
        chapter = database.Chapter(
            book_id=book.id,
            number=ch.number,
            title=ch.title,
            raw_text=ch.text,
            word_count=ch.word_count,
        )
        db.add(chapter)

    db.commit()
    db.refresh(book)
    return book


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/import", response_model=models.BookDetailResponse)
async def import_book(
    file: UploadFile = File(...),
    model_size: str | None = Form(None),
    narrator_voice_id: str | None = Form(None),
    db: Session = Depends(get_db),
) -> models.BookDetailResponse:
    """Upload an ebook, parse it, create Book + Chapters, return BookDetailResponse."""
    # Validate extension
    filename = file.filename or ""
    if "." not in filename:
        raise HTTPException(status_code=400, detail="请选择带有扩展名的小说文件")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"暂不支持 .{ext}，请上传 TXT、EPUB、FB2 或 PDF")

    # UploadFile already spools large uploads; copy it progressively instead
    # of allocating another 200 MB byte string in the application.
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp_path = tmp.name
        total = 0
        try:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(status_code=400, detail="小说文件不能超过 200 MB")
                tmp.write(chunk)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    try:
        if not total:
            raise HTTPException(status_code=400, detail="小说文件是空的，请选择包含正文的文件")
        parsed = ingestion.parse_book(tmp_path, ext)
        # Temporary storage must never become the public book/chapter name.
        temporary_title = Path(tmp_path).stem
        original_title = Path(filename.replace("\\", "/")).stem
        if parsed.title == temporary_title:
            parsed.title = original_title
        for chapter in parsed.chapters:
            if chapter.title and temporary_title in chapter.title:
                chapter.title = chapter.title.replace(temporary_title, original_title)
        if not parsed.chapters or not any(ch.text.strip() for ch in parsed.chapters):
            raise HTTPException(
                status_code=400, detail="未提取到小说正文。扫描版 PDF 请先进行文字识别（OCR），或上传 TXT/EPUB。"
            )
    except ingestion.IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    book = _persist_parsed_book(parsed, db)
    return _book_to_detail(book, db)


@router.get("", response_model=list[models.BookResponse])
async def list_books(db: Session = Depends(get_db)) -> list[models.BookResponse]:
    """List all books, newest first."""
    books = db.query(database.Book).order_by(database.Book.created_at.desc()).all()
    return [_book_to_response(b, db) for b in books]


@router.get("/{book_id}", response_model=models.BookDetailResponse)
async def get_book(
    book_id: str,
    db: Session = Depends(get_db),
) -> models.BookDetailResponse:
    """Return book detail (metadata + chapters)."""
    book = _get_book_or_404(book_id, db)
    return _book_to_detail(book, db)


@router.patch("/{book_id}", response_model=models.BookDetailResponse)
async def patch_book(
    book_id: str,
    data: dict[str, Any],
    db: Session = Depends(get_db),
) -> models.BookDetailResponse:
    """Update book metadata (title, author, cover_path)."""
    book = _get_book_or_404(book_id, db)

    allowed_fields = {"title", "author", "cover_path"}
    for field_name, value in data.items():
        if field_name in allowed_fields:
            setattr(book, field_name, value)

    db.commit()
    db.refresh(book)
    return _book_to_detail(book, db)


@router.delete("/{book_id}")
async def delete_book(
    book_id: str,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a book and all its related rows (cascade)."""
    book = _get_book_or_404(book_id, db)

    # Cascade-delete child rows manually (SQLite FK may not be enforced)
    chapters = db.query(database.Chapter).filter_by(book_id=book_id).all()
    for chapter in chapters:
        db.query(database.BookSegment).filter_by(chapter_id=chapter.id).delete()
    db.query(database.Chapter).filter_by(book_id=book_id).delete()
    db.query(database.BookCharacter).filter_by(book_id=book_id).delete()
    # NULL out VoiceProfile.book_id rather than deleting — auto-cast profiles
    # should survive as orphaned-from-book voices.
    db.query(database.VoiceProfile).filter_by(book_id=book_id).update({"book_id": None})
    db.delete(book)
    db.commit()

    return {"message": "Book deleted"}


# NOTE: POST /{book_id}/analyze was a 202 stub here in A6.
# B4 replaces it with the real endpoint in routes/book_analysis.py.
