# ruff: noqa: RUF001, RUF003
"""Tests for the ebook ingestion service (all four formats)."""

import pytest

from backend.services.ingestion import parse_book, IngestionError, ParsedBook


def test_txt_detects_encoding_and_splits_chapters(tmp_path):
    p = tmp_path / "book.txt"
    p.write_text("CHAPTER ONE\n\nHello world.\n\nCHAPTER TWO\n\nGoodbye.", encoding="utf-8")
    result = parse_book(str(p), "txt")
    assert isinstance(result, ParsedBook)
    assert len(result.chapters) >= 2
    assert "Hello world" in result.chapters[0].text


def test_txt_cp1251_round_trip(tmp_path):
    p = tmp_path / "ru.txt"
    p.write_bytes("Глава 1\n\nПривет.".encode("cp1251"))
    result = parse_book(str(p), "txt")
    assert "Привет" in result.chapters[0].text


def test_txt_word_count_populated(tmp_path):
    p = tmp_path / "wc.txt"
    p.write_text("CHAPTER ONE\n\nHello world foo bar.", encoding="utf-8")
    result = parse_book(str(p), "txt")
    assert result.chapters[0].word_count > 0


def test_txt_title_from_filename(tmp_path):
    p = tmp_path / "mybook.txt"
    p.write_text("Some content here.", encoding="utf-8")
    result = parse_book(str(p), "txt")
    assert result.title == "mybook"


def test_epub_metadata_and_chapters(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("Silo")
    book.add_author("Hugh Howey")
    c1 = epub.EpubHtml(title="Ch1", file_name="c1.xhtml")
    c1.content = b"<h1>Chapter 1</h1><p>It was a dark night.</p>"
    book.add_item(c1)
    book.spine = ["nav", c1]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "silo.epub"
    epub.write_epub(str(path), book)
    result = parse_book(str(path), "epub")
    assert result.title == "Silo"
    assert result.author == "Hugh Howey"
    assert any("dark night" in c.text for c in result.chapters)


def test_epub_chapter_word_count(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("Test")
    book.add_author("Author")
    c1 = epub.EpubHtml(title="Ch1", file_name="c1.xhtml")
    c1.content = b"<p>one two three four five.</p>"
    book.add_item(c1)
    book.spine = ["nav", c1]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "test.epub"
    epub.write_epub(str(path), book)
    result = parse_book(str(path), "epub")
    assert result.chapters[0].word_count >= 5


def test_fb2_em_dash_text_extracted(tmp_path):
    fb2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
 <description><title-info>
   <book-title>Test FB2</book-title>
   <author><first-name>Ivan</first-name><last-name>Petrov</last-name></author>
 </title-info></description>
 <body><section><title><p>Chapter 1</p></title>
   <p>— Hello, said the man.</p></section></body>
</FictionBook>"""
    p = tmp_path / "t.fb2"
    p.write_text(fb2, encoding="utf-8")
    result = parse_book(str(p), "fb2")
    assert result.title == "Test FB2"
    assert "Hello" in result.chapters[0].text


def test_fb2_author_combined(tmp_path):
    fb2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
 <description><title-info>
   <book-title>My Novel</book-title>
   <author><first-name>Jane</first-name><last-name>Doe</last-name></author>
 </title-info></description>
 <body><section><p>Hello world.</p></section></body>
</FictionBook>"""
    p = tmp_path / "novel.fb2"
    p.write_text(fb2, encoding="utf-8")
    result = parse_book(str(p), "fb2")
    assert result.author == "Jane Doe"


def test_fb2_multiple_sections(tmp_path):
    fb2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
 <description><title-info>
   <book-title>Multi</book-title>
   <author><first-name>A</first-name><last-name>B</last-name></author>
 </title-info></description>
 <body>
   <section><title><p>Part 1</p></title><p>First section text.</p></section>
   <section><title><p>Part 2</p></title><p>Second section text.</p></section>
 </body>
</FictionBook>"""
    p = tmp_path / "multi.fb2"
    p.write_text(fb2, encoding="utf-8")
    result = parse_book(str(p), "fb2")
    assert len(result.chapters) == 2


def test_pdf_best_effort(tmp_path):
    """Build a one-page PDF with fitz and verify text is extracted and de-hyphenated."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "CHAPTER ONE\n\nThis is an exam-\nple sentence.", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = parse_book(str(pdf_path), "pdf")
    assert isinstance(result, ParsedBook)
    # De-hyphenation: "exam-\nple" -> "example"
    full_text = " ".join(c.text for c in result.chapters)
    assert "example" in full_text


def test_pdf_empty_scanned_does_not_crash(tmp_path):
    """An image-only PDF with no text layer should not crash — return empty chapters."""
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank page, no text
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Must not raise
    result = parse_book(str(pdf_path), "pdf")
    assert isinstance(result, ParsedBook)
    assert result.source_format == "pdf"


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "bad.epub"
    p.write_bytes(b"not a zip")
    with pytest.raises(IngestionError):
        parse_book(str(p), "epub")


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "book.doc"
    p.write_bytes(b"some data")
    with pytest.raises(IngestionError, match="Unsupported"):
        parse_book(str(p), "doc")


def test_format_with_dot_prefix(tmp_path):
    """Format string may include leading dot (e.g. '.txt')."""
    p = tmp_path / "book.txt"
    p.write_text("Hello world.", encoding="utf-8")
    result = parse_book(str(p), ".txt")
    assert isinstance(result, ParsedBook)


def test_txt_no_chapter_headings_single_chapter(tmp_path):
    """A TXT file with no chapter headings should produce at least one chapter."""
    p = tmp_path / "prose.txt"
    p.write_text("Just some prose without headings.", encoding="utf-8")
    result = parse_book(str(p), "txt")
    assert len(result.chapters) >= 1
    assert "prose" in result.chapters[0].text


# ---------------------------------------------------------------------------
# Fix 1: XXE hardening in FB2 parser
# ---------------------------------------------------------------------------


def test_fb2_xxe_entity_not_expanded(tmp_path):
    """FB2 with an internal entity declaration must not expand it (XXE hardening).

    The parser must either raise IngestionError or return text that does NOT
    contain the expanded entity value — e.g. the /etc/passwd content must not
    appear in the output.
    """
    # An internal entity that would expand to a sensitive string
    xxe_fb2 = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!DOCTYPE FictionBook [\n"
        '  <!ENTITY xxe "SECRET_EXPANDED_CONTENT">\n'
        "]>\n"
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">\n'
        " <description><title-info>\n"
        "   <book-title>XXE Test</book-title>\n"
        " </title-info></description>\n"
        " <body><section><p>&xxe;</p></section></body>\n"
        "</FictionBook>"
    )
    p = tmp_path / "xxe.fb2"
    p.write_text(xxe_fb2, encoding="utf-8")

    # The hardened parser must NOT expand the entity — either raise IngestionError
    # or return text that does not contain the expanded entity value.
    raised = None
    result = None
    try:
        result = parse_book(str(p), "fb2")
    except IngestionError as exc:
        raised = exc  # parse failure is an acceptable safe outcome

    if raised is not None:
        # IngestionError is an acceptable safe outcome — entity processing blocked
        pass
    else:
        # No exception — verify entity was NOT silently expanded
        full_text = " ".join(c.text for c in result.chapters)
        assert "SECRET_EXPANDED_CONTENT" not in full_text, "Entity was expanded — XXE hardening not applied"


# ---------------------------------------------------------------------------
# Fix 2: EPUB nav/TOC must not appear as a chapter
# ---------------------------------------------------------------------------


def test_epub_nav_not_included_as_chapter(tmp_path):
    """The EPUB NAV document must be filtered out and not produce a chapter."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("Silo")
    book.add_author("Hugh Howey")

    c1 = epub.EpubHtml(title="Ch1", file_name="c1.xhtml")
    c1.content = b"<h1>Chapter 1</h1><p>It was a dark night.</p>"
    book.add_item(c1)

    book.spine = ["nav", c1]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    path = tmp_path / "silo_nav.epub"
    epub.write_epub(str(path), book)

    result = parse_book(str(path), "epub")

    # Should have exactly 1 real content chapter (not 2 due to nav)
    assert len(result.chapters) == 1, (
        f"Expected 1 chapter but got {len(result.chapters)}: {[c.title for c in result.chapters]}"
    )
    # The chapter should contain the real content, not nav boilerplate
    assert "dark night" in result.chapters[0].text
    # No chapter should be solely the book title (which nav often renders)
    for chapter in result.chapters:
        stripped = chapter.text.strip()
        assert stripped != "Silo", f"Chapter text is just the title — likely the nav doc leaked in: {stripped!r}"


# ---------------------------------------------------------------------------
# Fix 3: Multi-page PDF header/footer stripping
# ---------------------------------------------------------------------------


def test_pdf_header_footer_stripped(tmp_path):
    """Running header on every page must be stripped; body text must survive.

    Each page has the header at a fixed y-position and body text at a deliberately
    different y-position so only the header y is counted as repeated (≥50% of pages).
    """
    import fitz

    HEADER = "Running Header"
    # Body text for pages 1-3 inserted at DIFFERENT y-positions so the
    # body-text y-coords are page-unique (appear on only 1/3 pages) and thus
    # survive the Counter-based filter, while the header y appears on all pages.
    BODY_TEXTS = [
        ("First page body text.", 100),  # page 1 body at y=100
        ("Second page body text.", 200),  # page 2 body at y=200
        ("Third page body text.", 300),  # page 3 body at y=300
    ]

    doc = fitz.open()
    for body_text, body_y in BODY_TEXTS:
        page = doc.new_page()
        # Repeating header at the same y on every page
        page.insert_text((72, 36), HEADER, fontsize=10)
        # Body at a unique y-position per page so it is NOT filtered out
        page.insert_text((72, body_y), body_text, fontsize=12)
    pdf_path = tmp_path / "multipage.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = parse_book(str(pdf_path), "pdf")
    full_text = " ".join(c.text for c in result.chapters)

    # Header must have been stripped (it appeared on all 3 pages at same y)
    assert HEADER not in full_text, f"Repeated header {HEADER!r} was NOT stripped from output"
    # Body content must still be present (each at a unique y-position, count=1/3)
    for body_text, _ in BODY_TEXTS:
        keyword = body_text.split()[0]  # "First", "Second", "Third"
        assert keyword in full_text, f"Body keyword {keyword!r} missing — body text was incorrectly stripped"


def test_chinese_headings_front_matter_and_gb18030_are_preserved(tmp_path):
    from backend.services.ingestion import count_text_units

    text = "这是前言。\n第一章 风起\n林默说：“你好。”\n第２章 重逢\n三十年后，他们重逢。"
    path = tmp_path / "长夜.txt"
    path.write_bytes(text.encode("gb18030"))
    parsed = parse_book(str(path), "txt")
    assert len(parsed.chapters) == 3
    assert "第一章 风起" in parsed.chapters[1].text
    assert "第２章 重逢" in parsed.chapters[2].text
    assert "".join(c.text for c in parsed.chapters).replace("\n", "") == text.replace("\n", "")
    assert count_text_units("长夜漫漫 hello world") == 6


def test_giant_chinese_line_is_automatically_fragmented_without_truncation(tmp_path):
    from backend.services.ingestion import MAX_CHAPTER_CHARS

    text = "林默在山谷中穿行。" * 7000
    path = tmp_path / "大部头.txt"
    path.write_text(text, encoding="utf-8")
    result = parse_book(str(path), "txt")
    assert len(result.chapters) > 2
    assert max(len(c.text) for c in result.chapters) <= MAX_CHAPTER_CHARS
    assert "".join(c.text for c in result.chapters) == text
    assert len({c.title for c in result.chapters}) == len(result.chapters)


@pytest.mark.parametrize(
    "text,language",
    [
        ("林默说，明天我们还会再见。", "zh"),
        ("明日はまた会いましょう。", "ja"),
        ("We will meet again tomorrow.", "en"),
        ("Мы встретимся завтра.", "ru"),
        ("내일 다시 만나요.", "ko"),
    ],
)
def test_text_language_uses_novel_language(text, language):
    from backend.services.ingestion import detect_text_language

    assert detect_text_language(text) == language
