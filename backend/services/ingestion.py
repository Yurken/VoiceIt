# ruff: noqa: RUF001
"""Ebook ingestion — format-dispatched parsers producing normalized chapters.

Supports: EPUB (via ebooklib), FB2 (via lxml), TXT (via chardet), PDF (via PyMuPDF).
PDF parsing is best-effort: strips repeated headers/footers and de-hyphenates.
Does NOT call any LLM or TTS service — pure synchronous parsing only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

# Hardened XML parser: disallows entity resolution, DTD loading, and network access.
# Mitigates XXE (XML External Entity) file-disclosure and SSRF attacks when parsing
# untrusted uploaded ebook files.
_FB2_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    huge_tree=False,
)


class IngestionError(Exception):
    """Raised for unsupported or corrupt input files."""


@dataclass
class ParsedChapter:
    number: int
    title: str | None
    text: str
    word_count: int = 0


@dataclass
class ParsedBook:
    title: str
    author: str | None
    source_format: str
    chapters: list[ParsedChapter] = field(default_factory=list)
    cover_bytes: bytes | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_book(path: str, source_format: str) -> ParsedBook:
    """Dispatch to format-specific parser and return a normalised ParsedBook.

    Args:
        path: Absolute or relative filesystem path to the ebook file.
        source_format: One of 'epub', 'fb2', 'txt', 'pdf' (leading dot OK, e.g. '.epub').

    Returns:
        ParsedBook with chapters, metadata, and optional cover bytes.

    Raises:
        IngestionError: If the format is unsupported or the file is corrupt/unreadable.
    """
    fmt = source_format.lower().lstrip(".")
    parser = {"epub": _parse_epub, "fb2": _parse_fb2, "txt": _parse_txt, "pdf": _parse_pdf}.get(fmt)
    if parser is None:
        raise IngestionError(f"Unsupported format: {source_format!r}")
    try:
        parsed = parser(path)
        # Apply the same hard chapter bound to every supported format.
        bounded: list[ParsedChapter] = []
        for chapter in parsed.chapters:
            _append_chapter(bounded, chapter.title, chapter.text)
        parsed.chapters = bounded
        return parsed
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to parse {fmt!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Chinese novels frequently have no spaces or blank lines between chapters.
_CHAPTER_RE = re.compile(
    r"^[ \t\u3000]*(?:(?:chapter|глава|part|часть)\b[^\n]{0,110}|"
    r"第[零〇一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟0-9０-９]+[章回节卷部篇][^\n]{0,90}|"
    r"(?:序章|楔子|序言|前言|尾声|后记|终章|番外)(?:[ \t：:、.][^\n]{0,90})?)[ \t\u3000]*$",
    re.IGNORECASE | re.MULTILINE,
)
MAX_CHAPTER_CHARS = 20_000


def count_text_units(text: str) -> int:
    """Chinese characters plus space-delimited words, for useful word counts."""
    return len(
        re.findall(
            r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[A-Za-z0-9\u00c0-\u02af\u0400-\u04ff]+(?:['’-][A-Za-z0-9\u00c0-\u02af\u0400-\u04ff]+)*",
            text,
        )
    )


def detect_text_language(text: str) -> str:
    """Small deterministic ISO language detector shared by analysis and TTS."""
    sample = text[:16000]
    cjk = len(re.findall(r"[\u3400-\u9fff]", sample))
    kana = len(re.findall(r"[\u3040-\u30ff]", sample))
    hangul = len(re.findall(r"[\uac00-\ud7af]", sample))
    cyrillic = len(re.findall(r"[\u0400-\u04ff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if kana >= 2 and kana * 4 >= cjk:
        return "ja"
    if hangul >= 2 and hangul > max(cjk, latin // 2):
        return "ko"
    if cjk >= 2 and cjk * 3 >= latin:
        return "zh"
    if cyrillic >= 2 and cyrillic > latin:
        return "ru"
    return "en"


def _append_chapter(chapters: list[ParsedChapter], title: str | None, body: str) -> None:
    """Automatically fragment a large natural chapter, retaining every character."""
    body = _clean(body)
    if not body:
        return
    parts = []
    while len(body) > MAX_CHAPTER_CHARS:
        window = body[:MAX_CHAPTER_CHARS]
        boundaries = [m.end() for m in re.finditer(r"\n|[。！？!?]", window) if m.end() >= MAX_CHAPTER_CHARS // 2]
        end = boundaries[-1] if boundaries else MAX_CHAPTER_CHARS
        parts.append(body[:end].strip())
        body = body[end:].lstrip()
    if body:
        parts.append(body)
    for part_idx, part in enumerate(parts):
        part_title = title
        if len(parts) > 1:
            part_title = f"{title or '正文'}（{part_idx + 1}/{len(parts)}）"
        chapters.append(
            ParsedChapter(number=len(chapters) + 1, title=part_title, text=part, word_count=count_text_units(part))
        )


def _split_into_chapters(text: str, default_title: str | None = None) -> list[ParsedChapter]:
    """Keep front matter and heading text; impose a hard upper bound per part."""
    matches = list(_CHAPTER_RE.finditer(text))
    chapters: list[ParsedChapter] = []
    if not matches:
        _append_chapter(chapters, default_title, text)
        return chapters
    _append_chapter(chapters, "序言", text[: matches[0].start()])
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        _append_chapter(chapters, match.group(0).strip(), text[match.start() : end])
    return chapters


def _clean(text: str) -> str:
    """Collapse excess blank lines and strip leading/trailing whitespace."""
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html(html_bytes: bytes | str) -> str:
    """Strip HTML tags from *html_bytes*, returning plain text."""
    if isinstance(html_bytes, bytes):
        html_str = html_bytes.decode("utf-8", errors="replace")
    else:
        html_str = html_bytes
    # Remove script/style blocks
    html_str = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html_str, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines for paragraph separation
    html_str = re.sub(r"</(p|div|h[1-6]|li|tr|br)[^>]*>", "\n", html_str, flags=re.IGNORECASE)
    html_str = re.sub(r"<br\s*/?>", "\n", html_str, flags=re.IGNORECASE)
    # Strip all remaining tags
    html_str = re.sub(r"<[^>]+>", "", html_str)
    # Decode common HTML entities
    html_str = (
        html_str.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
    )
    return html_str


# ---------------------------------------------------------------------------
# EPUB parser
# ---------------------------------------------------------------------------


def _parse_epub(path: str) -> ParsedBook:
    import ebooklib
    from ebooklib import epub

    try:
        book = epub.read_epub(path, options={"ignore_ncx": True})
    except Exception as exc:
        raise IngestionError(f"Cannot open EPUB: {exc}") from exc

    # Metadata
    title_meta = book.get_metadata("DC", "title")
    title = title_meta[0][0] if title_meta else Path(path).stem

    creator_meta = book.get_metadata("DC", "creator")
    author = creator_meta[0][0] if creator_meta else None

    # Cover image
    cover_bytes: bytes | None = None
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_COVER:
            cover_bytes = item.get_content()
            break
    if cover_bytes is None:
        cover_item = book.get_item_with_id("cover")
        if cover_item is not None:
            cover_bytes = cover_item.get_content()

    # Chapters — iterate document items, skipping navigation documents.
    # EpubNav items have get_type() == ITEM_NAVIGATION; however, ebooklib sometimes
    # reports them as ITEM_DOCUMENT after a round-trip.  Filtering by the class type
    # or by file name containing "nav" is more reliable.
    from ebooklib.epub import EpubNav

    chapters: list[ParsedChapter] = []
    chapter_num = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        # Skip navigation/TOC documents — they are structural metadata, not content.
        if isinstance(item, EpubNav):
            continue
        if item.get_type() == ebooklib.ITEM_NAVIGATION:
            continue
        # Guard against nav items that survived the class/type check (e.g. name heuristic)
        item_name = item.get_name().lower()
        if "nav" in item_name and item_name.endswith((".xhtml", ".html", ".htm")):
            continue

        raw_content = item.get_content()
        if not raw_content:
            continue
        text = _strip_html(raw_content)
        text = _clean(text)
        if not text:
            continue
        chapter_num += 1
        chapters.append(
            ParsedChapter(
                number=chapter_num,
                title=item.get_name(),
                text=text,
                word_count=count_text_units(text),
            )
        )

    return ParsedBook(
        title=title,
        author=author,
        source_format="epub",
        chapters=chapters,
        cover_bytes=cover_bytes,
    )


# ---------------------------------------------------------------------------
# FB2 parser
# ---------------------------------------------------------------------------

_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"


def _fb2_text(elem) -> str:
    """Recursively collect text from an lxml element."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_fb2_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p for p in parts if p.strip())


def _parse_fb2(path: str) -> ParsedBook:
    try:
        tree = etree.parse(path, _FB2_PARSER)
    except Exception as exc:
        raise IngestionError(f"Cannot parse FB2 XML: {exc}") from exc

    root = tree.getroot()
    ns = _FB2_NS

    # Metadata
    title_el = root.find(f".//{{{ns}}}book-title")
    title = title_el.text.strip() if title_el is not None and title_el.text else Path(path).stem

    first_name_el = root.find(f".//{{{ns}}}first-name")
    last_name_el = root.find(f".//{{{ns}}}last-name")
    parts = []
    if first_name_el is not None and first_name_el.text:
        parts.append(first_name_el.text.strip())
    if last_name_el is not None and last_name_el.text:
        parts.append(last_name_el.text.strip())
    author = " ".join(parts) if parts else None

    # Sections → chapters
    body = root.find(f"{{{ns}}}body")
    if body is None:
        # Fallback: try without namespace
        body = root.find("body")

    chapters: list[ParsedChapter] = []
    if body is not None:
        sections = body.findall(f"{{{ns}}}section")
        if not sections:
            sections = body.findall("section")
        for idx, section in enumerate(sections):
            # Extract title if present
            title_el2 = section.find(f"{{{ns}}}title")
            chapter_title: str | None = None
            if title_el2 is not None:
                chapter_title = _fb2_text(title_el2).strip() or None

            # All <p> elements (recursively, to handle nested sections)
            paragraphs = section.findall(f".//{{{ns}}}p")
            if not paragraphs:
                paragraphs = section.findall(".//p")
            lines = []
            for p in paragraphs:
                ptext = _fb2_text(p).strip()
                if ptext:
                    lines.append(ptext)
            text = _clean("\n\n".join(lines))
            chapters.append(
                ParsedChapter(
                    number=idx + 1,
                    title=chapter_title,
                    text=text,
                    word_count=count_text_units(text),
                )
            )

    return ParsedBook(
        title=title,
        author=author,
        source_format="fb2",
        chapters=chapters,
    )


# ---------------------------------------------------------------------------
# TXT parser
# ---------------------------------------------------------------------------


def _parse_txt(path: str) -> ParsedBook:
    import codecs

    import chardet

    with Path(path).open("rb") as stream:
        sample = stream.read(1_048_576)
    detected = chardet.detect(sample).get("encoding") or "utf-8"
    if detected.lower().replace("-", "") in {"gb2312", "gbk"}:
        detected = "gb18030"
    # Prefer valid UTF-8 over a heuristic encoding guess, including a BOM.
    try:
        codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
        encodings = ["utf-8-sig", detected, "gb18030"]
    except UnicodeDecodeError:
        encodings = [detected, "gb18030", "utf-8-sig"]
    title = Path(path).stem
    for encoding in dict.fromkeys(encodings):
        chapters: list[ParsedChapter] = []
        parts: list[str] = []
        current_title: str | None = title
        part_size = 0
        try:
            # Decode progressively; a giant text file never needs a second full
            # raw-byte copy, and oversized chapters flush into bounded parts.
            with Path(path).open("r", encoding=encoding, newline=None) as stream:
                for line in stream:
                    if _CHAPTER_RE.fullmatch(line.rstrip("\r\n")):
                        _append_chapter(chapters, current_title, "".join(parts))
                        parts, part_size = [], 0
                        current_title = line.strip()
                    # A single unbroken line may itself be larger than a part.
                    for offset in range(0, len(line), MAX_CHAPTER_CHARS):
                        piece = line[offset : offset + MAX_CHAPTER_CHARS]
                        if part_size + len(piece) > MAX_CHAPTER_CHARS and parts:
                            _append_chapter(chapters, current_title, "".join(parts))
                            parts, part_size = [], 0
                        parts.append(piece)
                        part_size += len(piece)
            _append_chapter(chapters, current_title, "".join(parts))
            # Repeated fragments of one source chapter remain distinguishable.
            title_counts: dict[str | None, int] = {}
            for chapter in chapters:
                title_counts[chapter.title] = title_counts.get(chapter.title, 0) + 1
            title_indexes: dict[str | None, int] = {}
            for chapter in chapters:
                natural_title = chapter.title
                if title_counts[natural_title] > 1:
                    title_indexes[natural_title] = title_indexes.get(natural_title, 0) + 1
                    chapter.title = (
                        f"{natural_title or '正文'}（{title_indexes[natural_title]}/{title_counts[natural_title]}）"
                    )
            return ParsedBook(title=title, author=None, source_format="txt", chapters=chapters)
        except (UnicodeDecodeError, LookupError):
            continue
    raise IngestionError("无法识别小说文本编码，请将文件另存为 UTF-8 后上传。")


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------


def _parse_pdf(path: str) -> ParsedBook:
    """Best-effort PDF text extraction with header/footer stripping and de-hyphenation."""
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise IngestionError(f"Cannot open PDF: {exc}") from exc

    title = Path(path).stem
    num_pages = len(doc)

    if num_pages == 0:
        doc.close()
        return ParsedBook(title=title, author=None, source_format="pdf")

    # Collect per-page lines with their y-coordinates (rounded to nearest pt)
    page_lines: list[list[tuple[int, str]]] = []  # per page: list of (y_rounded, line_text)
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        lines_on_page: list[tuple[int, str]] = []
        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                y = round(line["bbox"][1])  # top-y coordinate rounded to int
                line_text = " ".join(span["text"] for span in line.get("spans", [])).strip()
                if line_text:
                    lines_on_page.append((y, line_text))
        page_lines.append(lines_on_page)

    doc.close()

    # Identify repeated y-positions (header/footer heuristic: same y on ≥50% of pages)
    if num_pages >= 2:
        from collections import Counter

        y_counts: Counter[int] = Counter()
        for lines in page_lines:
            for y, _ in lines:
                y_counts[y] += 1
        threshold = num_pages * 0.5
        repeated_ys = {y for y, count in y_counts.items() if count >= threshold}
    else:
        repeated_ys: set[int] = set()

    # Build full text with header/footer lines removed
    page_texts: list[str] = []
    for lines in page_lines:
        filtered = [line for y, line in lines if y not in repeated_ys]
        page_texts.append("\n".join(filtered))

    full_text = "\n\n".join(page_texts)

    # De-hyphenate: "exam-\nple" → "example"
    full_text = re.sub(r"-\n(\w)", r"\1", full_text)

    chapters = _split_into_chapters(full_text, default_title=title)

    return ParsedBook(
        title=title,
        author=None,
        source_format="pdf",
        chapters=chapters,
    )
