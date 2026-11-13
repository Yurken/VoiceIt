"""Tests for audiobook_export service.

Runs on synthetic short WAV clips — no GPU/model required.
Tests all three output formats (m4b, mp3_single, mp3_per_chapter) and asserts
M4B chapter markers via ffprobe.
"""

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers for creating synthetic audio
# ---------------------------------------------------------------------------


def _make_sine_wav(path: str, freq: float = 440.0, duration: float = 0.5, sr: int = 24000) -> None:
    """Write a short sine-wave WAV to *path*."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr)


def _make_chapter(tmp_path: Path, chapter_num: int, n_segments: int = 3) -> dict:
    """Create a chapter dict with *n_segments* synthetic WAV files."""
    seg_paths = []
    for i in range(n_segments):
        p = str(tmp_path / f"ch{chapter_num}_seg{i}.wav")
        _make_sine_wav(p, freq=220 + chapter_num * 100 + i * 20)
        seg_paths.append(p)
    return {
        "number": chapter_num,
        "title": f"Chapter {chapter_num}",
        "segment_paths": seg_paths,
    }


def _probe_audio(path: str) -> dict:
    """Inspect the produced artifact through FFmpeg's public metadata view."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _audio_packet_hash(path: str) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-c:a", "copy",
         "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Import the service under test (will fail RED before implementation exists)
# ---------------------------------------------------------------------------


from backend.services.audiobook_export import export_book, read_chapter_markers  # noqa: E402
from backend.utils.audio import normalize_loudness  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_chapter_book(tmp_path):
    """Return (chapters, options_base) for a two-chapter synthetic book."""
    chapters = [
        _make_chapter(tmp_path, chapter_num=1, n_segments=3),
        _make_chapter(tmp_path, chapter_num=2, n_segments=2),
    ]
    return chapters, tmp_path


@pytest.fixture()
def cover_image(tmp_path):
    """Create a minimal PNG (1x1 pixel) and return its path."""
    # Minimal valid PNG bytes (1x1 transparent pixel)
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    cover = tmp_path / "cover.png"
    cover.write_bytes(png_bytes)
    return str(cover)


# ---------------------------------------------------------------------------
# normalize_loudness unit tests
# ---------------------------------------------------------------------------


class TestNormalizeLoudness:
    def test_returns_numpy_array(self):
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        result = normalize_loudness(audio, sample_rate=24000, target_lufs=-18.0)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = normalize_loudness(audio, sample_rate=24000, target_lufs=-18.0)
        assert result.shape == audio.shape

    def test_silent_audio_returned_unchanged(self):
        """Completely silent audio should not blow up."""
        audio = np.zeros(24000, dtype=np.float32)
        result = normalize_loudness(audio, sample_rate=24000, target_lufs=-18.0)
        assert result is not None
        assert result.shape == audio.shape

    def test_loudness_is_closer_to_target(self):
        """After normalization the loudness should be within 2 LU of target."""
        import pyloudnorm as pyln

        sr = 24000
        # Create 1-second pink-ish signal that pyloudnorm can measure
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        meter = pyln.Meter(sr)
        target = -18.0
        result = normalize_loudness(audio, sample_rate=sr, target_lufs=target)
        measured = meter.integrated_loudness(result)
        # If signal was measurable, check closeness
        if not np.isinf(measured):
            assert abs(measured - target) < 3.0


# ---------------------------------------------------------------------------
# export_book — M4B format
# ---------------------------------------------------------------------------


class TestExportM4B:
    def test_produces_m4b_file(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, filename = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "m4b",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        assert Path(out_path).exists(), "Output file should exist"
        assert out_path.endswith(".m4b"), "Output should be .m4b"
        assert filename.endswith(".m4b")

    def test_m4b_has_chapter_markers(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "m4b",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        markers = read_chapter_markers(out_path)
        # Should have one marker per chapter
        assert len(markers) == len(chapters), (
            f"Expected {len(chapters)} chapter markers, got {len(markers)}: {markers}"
        )

    def test_m4b_chapter_titles(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "m4b",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        markers = read_chapter_markers(out_path)
        titles = [m["title"] for m in markers]
        assert "Chapter 1" in titles
        assert "Chapter 2" in titles

    def test_m4b_with_cover(self, two_chapter_book, cover_image, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "m4b",
                "title": "Test Book",
                "author": "Test Author",
                "cover_path": cover_image,
            },
        )
        assert Path(out_path).exists()
        assert any(stream.get("disposition", {}).get("attached_pic") == 1 for stream in _probe_audio(out_path)["streams"])


# ---------------------------------------------------------------------------
# export_book — MP3 single format
# ---------------------------------------------------------------------------


class TestExportMP3Single:
    def test_produces_mp3_file(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, filename = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_single",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        assert Path(out_path).exists()
        assert out_path.endswith(".mp3")
        assert filename.endswith(".mp3")

    def test_mp3_has_id3_tags(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_single",
                "title": "My Audiobook",
                "author": "Test Author",
            },
        )
        tags = _probe_audio(out_path)["format"]["tags"]
        assert tags["title"] == "My Audiobook"
        assert tags["album"] == "My Audiobook"
        assert tags["artist"] == "Test Author"

    def test_mp3_with_cover(self, two_chapter_book, cover_image, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_single",
                "title": "Test Book",
                "author": "Test Author",
                "cover_path": cover_image,
            },
        )
        streams = _probe_audio(out_path)["streams"]
        assert any(stream.get("disposition", {}).get("attached_pic") == 1 for stream in streams)



# ---------------------------------------------------------------------------
# export_book — MP3 per-chapter ZIP format
# ---------------------------------------------------------------------------


class TestExportMP3PerChapter:
    def test_produces_zip_file(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, filename = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_per_chapter",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        assert Path(out_path).exists()
        assert out_path.endswith(".zip")
        assert filename.endswith(".zip")

    def test_zip_contains_one_mp3_per_chapter(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_per_chapter",
                "title": "Test Book",
                "author": "Test Author",
            },
        )
        with zipfile.ZipFile(out_path) as zf:
            mp3_files = [n for n in zf.namelist() if n.endswith(".mp3")]
        assert len(mp3_files) == len(chapters), (
            f"Expected {len(chapters)} MP3 files, got {len(mp3_files)}"
        )

    def test_zip_mp3s_have_id3_tags(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_per_chapter",
                "title": "My Book",
                "author": "Someone",
            },
        )
        with zipfile.ZipFile(out_path) as zf:
            mp3_names = sorted(n for n in zf.namelist() if n.endswith(".mp3"))
            assert len(mp3_names) >= 1
            # Extract first MP3 and check its tags
            with tempfile.TemporaryDirectory() as td:
                zf.extract(mp3_names[0], td)
                mp3_path = os.path.join(td, mp3_names[0])
                tags = _probe_audio(mp3_path)["format"]["tags"]
                assert tags["title"] == "Chapter 1"
                assert tags["album"] == "My Book"
                assert tags["artist"] == "Someone"
                assert tags["track"] == "1"

    def test_zip_with_cover(self, two_chapter_book, cover_image, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={
                "format": "mp3_per_chapter",
                "title": "Test Book",
                "author": "Test Author",
                "cover_path": cover_image,
            },
        )
        assert Path(out_path).exists()
        with zipfile.ZipFile(out_path) as archive, tempfile.TemporaryDirectory() as directory:
            for name in archive.namelist():
                extracted = archive.extract(name, directory)
                streams = _probe_audio(extracted)["streams"]
                assert any(stream.get("disposition", {}).get("attached_pic") == 1 for stream in streams)



# ---------------------------------------------------------------------------
# read_chapter_markers tests
# ---------------------------------------------------------------------------


class TestReadChapterMarkers:
    def test_returns_list(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "m4b", "title": "X"},
        )
        markers = read_chapter_markers(out_path)
        assert isinstance(markers, list)

    def test_markers_have_title_and_start(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "m4b", "title": "X"},
        )
        markers = read_chapter_markers(out_path)
        for m in markers:
            assert "title" in m
            assert "start" in m

    def test_markers_start_times_ordered(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "m4b", "title": "X"},
        )
        markers = read_chapter_markers(out_path)
        starts = [m["start"] for m in markers]
        assert starts == sorted(starts), "Chapter start times must be in ascending order"


# ---------------------------------------------------------------------------
# Robustness: missing segments
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_missing_segment_skipped(self, tmp_path):
        """A chapter with a missing audio file should not abort the export."""
        good_wav = str(tmp_path / "good.wav")
        _make_sine_wav(good_wav)
        chapters = [
            {
                "number": 1,
                "title": "Chapter 1",
                "segment_paths": [good_wav, "/nonexistent/segment.wav", good_wav],
            }
        ]
        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "mp3_single", "title": "T"},
        )
        assert Path(out_path).exists()

    def test_all_segments_missing_raises(self, tmp_path):
        """If ALL segments are missing, export should raise an error."""
        chapters = [
            {
                "number": 1,
                "title": "Chapter 1",
                "segment_paths": ["/nonexistent/a.wav", "/nonexistent/b.wav"],
            }
        ]
        with pytest.raises(Exception):
            export_book(
                chapters=chapters,
                output_dir=str(tmp_path / "out"),
                options={"format": "mp3_single", "title": "T"},
            )

    def test_progress_callback_called(self, two_chapter_book, tmp_path):
        chapters, _ = two_chapter_book
        calls = []

        def cb(pct, msg=""):
            calls.append((pct, msg))

        export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "mp3_single", "title": "T"},
            progress_callback=cb,
        )
        assert len(calls) > 0, "Progress callback should have been called at least once"
        # Final call should be 100%
        assert calls[-1][0] == 100


# ---------------------------------------------------------------------------
# Fix 1: scene_pause_s is actually applied for scene-break segments
# ---------------------------------------------------------------------------


from backend.services.audiobook_export import _stitch_chapter  # noqa: E402


class TestSceneBreak:
    """Verify that is_scene_break markers cause scene_pause_s to be used."""

    def _make_wav(self, path: str, duration: float = 0.2, sr: int = 24000) -> None:
        _make_sine_wav(path, duration=duration, sr=sr)

    def test_scene_break_produces_longer_duration(self, tmp_path):
        """Chapter with a scene-break marker has a longer duration than without one."""
        seg_a = str(tmp_path / "a.wav")
        seg_b = str(tmp_path / "b.wav")
        seg_c = str(tmp_path / "c.wav")
        self._make_wav(seg_a)
        self._make_wav(seg_b)
        self._make_wav(seg_c)

        inter_pause = 0.1
        scene_pause = 1.0  # deliberately much larger

        # Without scene break: all pauses are inter_pause
        plain_paths = [seg_a, seg_b, seg_c]
        plain_audio = _stitch_chapter(
            plain_paths,
            inter_segment_pause_s=inter_pause,
            scene_pause_s=scene_pause,
        )

        # With a scene break before seg_b: pause before seg_b becomes scene_pause
        scene_paths = [
            seg_a,
            {"path": seg_b, "is_scene_break": True},
            seg_c,
        ]
        scene_audio = _stitch_chapter(
            scene_paths,
            inter_segment_pause_s=inter_pause,
            scene_pause_s=scene_pause,
        )

        assert plain_audio is not None
        assert scene_audio is not None

        # scene_audio should be longer by roughly (scene_pause - inter_pause) seconds
        plain_len = len(plain_audio)
        scene_len = len(scene_audio)
        assert scene_len > plain_len, (
            f"Scene-break audio ({scene_len} samples) should be longer than plain "
            f"audio ({plain_len} samples) when scene_pause > inter_segment_pause"
        )

    def test_no_scene_break_metadata_unchanged(self, tmp_path):
        """Plain string paths produce the same result as before (backward compat)."""
        seg_a = str(tmp_path / "a.wav")
        seg_b = str(tmp_path / "b.wav")
        self._make_wav(seg_a)
        self._make_wav(seg_b)

        inter_pause = 0.3
        plain = _stitch_chapter(
            [seg_a, seg_b],
            inter_segment_pause_s=inter_pause,
            scene_pause_s=0.8,
        )
        # Dict entries with is_scene_break=False should behave like plain strings
        dict_no_break = _stitch_chapter(
            [{"path": seg_a, "is_scene_break": False}, {"path": seg_b}],
            inter_segment_pause_s=inter_pause,
            scene_pause_s=0.8,
        )
        assert plain is not None
        assert dict_no_break is not None
        assert len(plain) == len(dict_no_break), (
            "Dict entries without is_scene_break should produce identical duration "
            "to plain string entries"
        )

    def test_scene_break_on_first_segment_ignored(self, tmp_path):
        """A scene-break flag on the first segment is silently ignored (no leading silence)."""
        seg_a = str(tmp_path / "a.wav")
        seg_b = str(tmp_path / "b.wav")
        self._make_wav(seg_a)
        self._make_wav(seg_b)

        # Mark the very first segment as a scene break — should behave identically
        # to not marking it (no leading silence added).
        first_break = _stitch_chapter(
            [{"path": seg_a, "is_scene_break": True}, seg_b],
            inter_segment_pause_s=0.1,
            scene_pause_s=1.0,
        )
        plain = _stitch_chapter(
            [seg_a, seg_b],
            inter_segment_pause_s=0.1,
            scene_pause_s=1.0,
        )
        assert first_break is not None
        assert plain is not None
        assert len(first_break) == len(plain), (
            "Scene-break on the first segment should not add leading silence"
        )


# ---------------------------------------------------------------------------
# Fix 2: M4B chapter markers are correct when a chapter is entirely skipped
# ---------------------------------------------------------------------------


class TestM4BSkippedChapter:
    """Verify that skipped chapters do not corrupt M4B chapter marker alignment."""

    def test_skipped_chapter_markers_aligned(self, tmp_path):
        """When chapter 2 is fully skipped (bad segments), chapters 1 and 3
        must appear with their correct titles in the M4B output — not shifted."""
        wav_a = str(tmp_path / "ch1_seg0.wav")
        wav_c = str(tmp_path / "ch3_seg0.wav")
        _make_sine_wav(wav_a, duration=0.5)
        _make_sine_wav(wav_c, freq=660.0, duration=0.5)

        chapters = [
            {
                "number": 1,
                "title": "The Beginning",
                "segment_paths": [wav_a],
            },
            {
                "number": 2,
                "title": "GHOST CHAPTER",
                "segment_paths": ["/nonexistent/bad.wav"],  # entirely missing
            },
            {
                "number": 3,
                "title": "The End",
                "segment_paths": [wav_c],
            },
        ]

        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "m4b", "title": "Skip Test"},
        )

        markers = read_chapter_markers(out_path)
        titles = [m["title"] for m in markers]

        # Only chapters 1 and 3 produced audio
        assert len(markers) == 2, f"Expected 2 markers (ch1+ch3), got {len(markers)}: {markers}"
        assert "The Beginning" in titles, f"'The Beginning' missing from markers: {titles}"
        assert "The End" in titles, f"'The End' missing from markers: {titles}"
        assert "GHOST CHAPTER" not in titles, (
            f"Skipped chapter should not appear in markers: {titles}"
        )

    def test_skipped_chapter_start_times_correct(self, tmp_path):
        """Chapter 1 must start at 0; chapter 3 must start strictly after ch1 ends."""
        wav_a = str(tmp_path / "ch1.wav")
        wav_c = str(tmp_path / "ch3.wav")
        _make_sine_wav(wav_a, duration=0.5)
        _make_sine_wav(wav_c, freq=660.0, duration=0.5)

        chapters = [
            {"number": 1, "title": "Alpha", "segment_paths": [wav_a]},
            {"number": 2, "title": "Ghost", "segment_paths": ["/nonexistent/x.wav"]},
            {"number": 3, "title": "Omega", "segment_paths": [wav_c]},
        ]

        out_path, _ = export_book(
            chapters=chapters,
            output_dir=str(tmp_path / "out"),
            options={"format": "m4b", "title": "Start Time Test"},
        )

        markers = read_chapter_markers(out_path)
        assert len(markers) == 2
        starts = [m["start"] for m in markers]
        assert starts[0] == pytest.approx(0.0, abs=0.05), (
            f"First marker should start near 0s, got {starts[0]}"
        )
        assert starts[1] > starts[0], (
            f"Second marker ({starts[1]}) should start after first ({starts[0]})"
        )


class TestStreamingExport:
    def test_staged_segments_are_released_instead_of_retaining_the_book(self, tmp_path, monkeypatch):
        import weakref
        import backend.services.audiobook_export as service

        chapters = [_make_chapter(tmp_path, number, n_segments=4) for number in range(1, 5)]
        original_load = service.load_audio
        loaded_arrays = []

        def tracked_load(*args, **kwargs):
            # A streaming writer may still hold the preceding segment at the
            # next read, but must not retain all previous segments or chapters.
            assert sum(reference() is not None for reference in loaded_arrays) <= 1
            audio, rate = original_load(*args, **kwargs)
            loaded_arrays.append(weakref.ref(audio))
            return audio, rate

        monkeypatch.setattr(service, "load_audio", tracked_load)
        out_dir = tmp_path / "out"
        output, _ = service.export_book(chapters, str(out_dir), {"format": "mp3_single", "title": "Bounded"})
        assert len(loaded_arrays) == 16
        assert all(reference() is None for reference in loaded_arrays)
        assert list(out_dir.iterdir()) == [Path(output)]

    def test_streamed_scene_pauses_chapter_pauses_and_markers_keep_exact_timing(self, tmp_path):
        segment = str(tmp_path / "short.wav")
        _make_sine_wav(segment, duration=0.2)
        chapters = [
            {"number": 1, "title": "Scene", "segment_paths": [segment, {"path": segment, "is_scene_break": True}, segment]},
            {"number": 2, "title": "After", "segment_paths": [segment, segment]},
        ]
        output, _ = export_book(chapters, str(tmp_path / "out"), {
            "format": "m4b", "title": "Timing", "inter_segment_pause_s": 0.15,
            "scene_pause_s": 0.7, "chapter_pause_s": 0.6,
        })
        markers = read_chapter_markers(output)
        assert markers[1]["start"] == pytest.approx(2.05, abs=0.01)
        assert float(_probe_audio(output)["format"]["duration"]) == pytest.approx(2.6, abs=0.08)

    def test_concat_manifest_handles_quotes_in_the_output_directory(self, tmp_path):
        chapter = _make_chapter(tmp_path, 1, n_segments=1)
        output, _ = export_book([chapter], str(tmp_path / "speaker's export"), {"format": "mp3_single", "title": "Quoted"})
        assert _probe_audio(output)["format"]["tags"]["title"] == "Quoted"


class TestFFmpegMP3Tags:
    def test_tagging_copies_audio_packets_and_preserves_unicode_metadata(self, tmp_path, cover_image):
        from backend.services.audiobook_export import _embed_mp3_tags, _encode_mp3

        wave = str(tmp_path / "voice.wav")
        output = str(tmp_path / "voice.mp3")
        _make_sine_wav(wave)
        _encode_mp3(wave, output)
        packet_hash = _audio_packet_hash(output)
        _embed_mp3_tags(output, title="夏夜的约定", album="多角色小说", artist="作者", track_num=7, cover_path=cover_image)
        assert _audio_packet_hash(output) == packet_hash
        probe = _probe_audio(output)
        assert probe["format"]["tags"]["title"] == "夏夜的约定"
        assert probe["format"]["tags"]["album"] == "多角色小说"
        assert probe["format"]["tags"]["artist"] == "作者"
        assert probe["format"]["tags"]["track"] == "7"
        assert any(stream.get("disposition", {}).get("attached_pic") == 1 for stream in probe["streams"])

    def test_remux_failure_leaves_original_intact_and_cleans_temporary_file(self, tmp_path, monkeypatch):
        import backend.services.audiobook_export as service

        original = tmp_path / "original.mp3"
        original.write_bytes(b"unchanged audio")

        def fail_remux(command):
            Path(command[-1]).write_bytes(b"incomplete temporary output")
            raise RuntimeError("tagging failed")

        monkeypatch.setattr(service, "_run_ffmpeg", fail_remux)
        with pytest.raises(RuntimeError, match="tagging failed"):
            service._embed_mp3_tags(str(original), title="Changed")
        assert original.read_bytes() == b"unchanged audio"
        assert list(tmp_path.iterdir()) == [original]


    def test_m4b_preserves_special_characters_in_book_and_chapter_metadata(self, tmp_path):
        chapter = _make_chapter(tmp_path, 1, n_segments=1)
        chapter["title"] = "雨夜; #1=灯光"
        output, _ = export_book([chapter], str(tmp_path / "out"), {"format": "m4b", "title": "夏夜; #序章=月光", "author": "作者; #某人"})
        tags = _probe_audio(output)["format"]["tags"]
        assert tags["title"] == "夏夜; #序章=月光"
        assert tags["artist"] == "作者; #某人"
        assert read_chapter_markers(output)[0]["title"] == chapter["title"]


    @pytest.mark.parametrize("output_format,extension", [("m4b", ".m4b"), ("mp3_single", ".mp3"), ("mp3_per_chapter", ".zip")])
    def test_failed_reexport_preserves_existing_complete_artifact(self, tmp_path, monkeypatch, output_format, extension):
        import backend.services.audiobook_export as service

        chapter = _make_chapter(tmp_path, 1, n_segments=1)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        existing = out_dir / f"Preserve{extension}"
        existing.write_bytes(b"previous complete audiobook")

        def fail_encode(*args, **kwargs):
            Path(args[1]).write_bytes(b"incomplete new encode")
            raise RuntimeError("encode interrupted")

        monkeypatch.setattr(service, "_encode_mp3", fail_encode)
        monkeypatch.setattr(service, "_encode_aac_m4b", fail_encode)
        with pytest.raises(RuntimeError, match="encode interrupted"):
            service.export_book([chapter], str(out_dir), {"format": output_format, "title": "Preserve"})
        assert existing.read_bytes() == b"previous complete audiobook"
        assert list(out_dir.iterdir()) == [existing]
