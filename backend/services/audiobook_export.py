"""
Audiobook export service.

Stitches a book's chapters (each described as an ordered list of audio segment
paths) with configurable inter-segment, scene, and chapter pauses, normalizes
the result to a target LUFS, then emits one of three formats:

  * ``m4b``            – single AAC-in-MP4 with FFmpeg chapter markers
  * ``mp3_single``     – single MP3 with ID3 tags (title/author/cover)
  * ``mp3_per_chapter``– ZIP archive containing one ID3-tagged MP3 per chapter

All heavy lifting (concatenation, encoding) is delegated to FFmpeg; Python only
streams one segment at a time to disk-backed chapter files, then FFmpeg applies
loudness normalization during encoding. Whole-book audio is never held in RAM.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from backend.utils.audio import load_audio

# ---------------------------------------------------------------------------
# Default pause durations
# ---------------------------------------------------------------------------

DEFAULT_INTER_SEGMENT_PAUSE_S: float = 0.3   # between adjacent segments
DEFAULT_SCENE_PAUSE_S: float = 0.8           # at detected scene/paragraph breaks
DEFAULT_CHAPTER_PAUSE_S: float = 1.5         # between chapters (appended at end)
DEFAULT_TARGET_LUFS: float = -18.0           # audiobook standard range −18 to −23
DEFAULT_BITRATE: str = "64k"
DEFAULT_SAMPLE_RATE: int = 24000
DEFAULT_CHANNELS: int = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_silence(duration_s: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Return a mono float32 silence array of *duration_s* seconds."""
    n_samples = max(1, int(sample_rate * duration_s))
    return np.zeros(n_samples, dtype=np.float32)


def _stitch_chapter(
    segment_paths: List[Union[str, Dict[str, Any]]],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    inter_segment_pause_s: float = DEFAULT_INTER_SEGMENT_PAUSE_S,
    scene_pause_s: float = DEFAULT_SCENE_PAUSE_S,
) -> Optional[np.ndarray]:
    """Load and concatenate all segment WAV files for one chapter.

    Missing or unreadable files are skipped gracefully.  Returns ``None`` if
    *all* segments fail to load.

    Each entry in *segment_paths* may be either:

    * a plain ``str`` path to a WAV file, **or**
    * a ``dict`` with at least a ``"path"`` key and an optional boolean
      ``"is_scene_break"`` key.  When ``is_scene_break`` is ``True``, the
      pause inserted *before* this segment (if it is not the first loaded
      segment) will be *scene_pause_s* instead of *inter_segment_pause_s*,
      allowing scene/paragraph breaks to have noticeably longer gaps.

    Backward-compatible: if all entries are plain strings, behavior is
    identical to the original implementation (inter-segment pauses only).
    """
    parts: List[np.ndarray] = []
    inter_silence = _make_silence(inter_segment_pause_s, sample_rate)
    scene_silence = _make_silence(scene_pause_s, sample_rate)

    loaded_count = 0  # number of segments successfully loaded so far

    for entry in segment_paths:
        # Normalize entry to (path, is_scene_break)
        if isinstance(entry, dict):
            path = entry["path"]
            is_scene_break = bool(entry.get("is_scene_break", False))
        else:
            path = entry
            is_scene_break = False

        try:
            audio, _ = load_audio(path, sample_rate=sample_rate, mono=True)
        except Exception:
            continue  # skip bad segments

        # Insert the appropriate pause before this segment (not before the first loaded one)
        if loaded_count > 0:
            if is_scene_break:
                parts.append(scene_silence)
            else:
                parts.append(inter_silence)

        parts.append(audio)
        loaded_count += 1

    if not parts:
        return None

    return np.concatenate(parts, axis=0).astype(np.float32)


def _write_wav_to_tmp(audio: np.ndarray, sample_rate: int) -> str:
    """Write *audio* to a temp WAV file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path


def _run_ffmpeg(cmd: List[str]) -> None:
    """Run an FFmpeg command; raise RuntimeError on failure."""
    # Avoid retaining FFmpeg's periodic status output for multi-hour books.
    command = [cmd[0], "-hide_banner", "-loglevel", "error", "-nostats", *cmd[1:]]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )


def _audio_input(path: str, concat: bool = False) -> List[str]:
    """Describe a single audio file or an ffconcat manifest without shell parsing."""
    return (["-f", "concat", "-safe", "0"] if concat else []) + ["-i", path]


def _normalization_args(target_lufs: Optional[float], sample_rate: int) -> List[str]:
    if target_lufs is None:
        return []
    # FFmpeg processes the stream with bounded filter buffers instead of loading
    # an entire multi-hour book into Python. Set the rate explicitly because
    # dynamic loudnorm internally upsamples to 192 kHz.
    return ["-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11", "-ar", str(sample_rate)]


def _encode_mp3(
    wav_path: str,
    out_path: str,
    bitrate: str = DEFAULT_BITRATE,
    channels: int = DEFAULT_CHANNELS,
    *,
    concat: bool = False,
    target_lufs: Optional[float] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Encode one file or a disk-backed book manifest to MP3."""
    cmd = ["ffmpeg", "-y"] + _audio_input(wav_path, concat)
    cmd += _normalization_args(target_lufs, sample_rate)
    cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate, "-ac", str(channels), out_path]
    _run_ffmpeg(cmd)


def _metadata_value(value: str) -> str:
    """Escape literal user text for FFmetadata, including chapter titles."""
    escaped = str(value).replace("\\", "\\\\")
    for character in ("=", ";", "#", "\n", "\r"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _encode_aac_m4b(
    wav_path: str,
    out_path: str,
    chapters: List[Dict[str, Any]],
    chapter_starts_s: List[float],
    title: str = "",
    author: str = "",
    cover_path: Optional[str] = None,
    bitrate: str = DEFAULT_BITRATE,
    channels: int = DEFAULT_CHANNELS,
    *,
    concat: bool = False,
    target_lufs: Optional[float] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    total_duration_s: Optional[float] = None,
) -> None:
    """Encode *wav_path* → M4B with embedded chapter markers via FFmpeg metadata."""
    # Build FFmpeg chapter metadata
    metadata_lines = [
        ";FFMETADATA1",
        f"title={_metadata_value(title)}",
        f"artist={_metadata_value(author)}",
        "",
    ]
    if total_duration_s is None:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", wav_path],
            capture_output=True,
            text=True,
            check=True,
        )
        total_duration_s = float(json.loads(result.stdout)["format"]["duration"])

    for i, (ch, start_s) in enumerate(zip(chapters, chapter_starts_s)):
        end_s = chapter_starts_s[i + 1] if i + 1 < len(chapter_starts_s) else total_duration_s
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        ch_title = ch.get("title", f"Chapter {ch.get('number', i + 1)}")
        metadata_lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_metadata_value(ch_title)}",
            "",
        ]

    fd, meta_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(metadata_lines))

        cmd = ["ffmpeg", "-y"] + _audio_input(wav_path, concat) + ["-i", meta_path]
        if cover_path and Path(cover_path).exists():
            cmd += ["-i", cover_path, "-map", "0:a:0", "-map", "2:v:0",
                    "-codec:v", "mjpeg", "-disposition:v:0", "attached_pic"]
        else:
            cmd += ["-map", "0:a:0"]
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]
        cmd += _normalization_args(target_lufs, sample_rate)
        cmd += ["-codec:a", "aac", "-b:a", bitrate, "-ac", str(channels), "-f", "ipod"]
        cmd.append(out_path)
        _run_ffmpeg(cmd)
    finally:
        Path(meta_path).unlink(missing_ok=True)


def _embed_mp3_tags(
    mp3_path: str,
    title: str = "",
    album: str = "",
    artist: str = "",
    track_num: Optional[int] = None,
    cover_path: Optional[str] = None,
) -> None:
    """Atomically remux MP3 ID3 tags and optional cover art using FFmpeg.

    Audio packets are copied without another lossy encode. The original file
    remains intact if FFmpeg fails, and the sibling temporary file is always
    removed. No optional Python tagging dependency is required.
    """
    source = Path(mp3_path)
    fd, tagged_path = tempfile.mkstemp(prefix=f".{source.stem}-tags-", suffix=".mp3", dir=source.parent)
    os.close(fd)
    try:
        cmd = ["ffmpeg", "-y", "-i", str(source)]
        has_cover = bool(cover_path and Path(cover_path).is_file())
        if has_cover:
            cmd += ["-i", str(cover_path), "-map", "0:a:0", "-map", "1:v:0"]
        else:
            # Preserve an existing attached picture when only tags are changed.
            cmd += ["-map", "0:a:0", "-map", "0:v?"]
        cmd += ["-map_metadata", "0", "-codec:a", "copy", "-codec:v", "copy",
                "-id3v2_version", "3"]
        for key, value in (("title", title), ("album", album), ("artist", artist)):
            if value:
                cmd += ["-metadata", f"{key}={value}"]
        if track_num is not None:
            cmd += ["-metadata", f"track={track_num}"]
        if has_cover:
            cmd += ["-disposition:v:0", "attached_pic", "-metadata:s:v:0", "title=Cover",
                    "-metadata:s:v:0", "comment=Cover (front)"]
        _run_ffmpeg(cmd + [tagged_path])
        os.replace(tagged_path, source)
    finally:
        Path(tagged_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public: read_chapter_markers
# ---------------------------------------------------------------------------


def read_chapter_markers(file_path: str) -> List[Dict[str, Any]]:
    """Extract chapter markers from an M4B (or any container) via ffprobe.

    Returns a list of dicts with keys ``title`` (str) and ``start`` (float,
    seconds).  Returns an empty list if the file has no chapter metadata or
    ffprobe fails.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_chapters",
            file_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    data = json.loads(result.stdout)
    chapters = data.get("chapters", [])
    markers = []
    for ch in chapters:
        title = ch.get("tags", {}).get("title", "")
        start = float(ch.get("start_time", 0))
        markers.append({"title": title, "start": start})
    return markers


def _write_chapter_wav(
    segment_paths: List[Union[str, Dict[str, Any]]],
    output_path: str,
    sample_rate: int,
    inter_segment_pause_s: float,
    scene_pause_s: float,
) -> int:
    """Write one segment at a time and return its exact PCM frame count.

    RF64 supports large chapters beyond the WAV 4 GiB limit. Only a single
    generated segment and the short pause buffers reside in Python memory.
    Unreadable segments are skipped just as in the original stitch helper.
    """
    inter_silence = _make_silence(inter_segment_pause_s, sample_rate)
    scene_silence = _make_silence(scene_pause_s, sample_rate)
    total_frames = 0
    with sf.SoundFile(output_path, "w", samplerate=sample_rate, channels=1,
                      format="RF64", subtype="PCM_16") as output:
        for entry in segment_paths:
            path = entry.get("path") if isinstance(entry, dict) else entry
            scene_break = bool(entry.get("is_scene_break", False)) if isinstance(entry, dict) else False
            try:
                audio, _ = load_audio(path, sample_rate=sample_rate, mono=True)
            except Exception:
                continue
            if not len(audio):
                continue
            if total_frames:
                pause = scene_silence if scene_break else inter_silence
                output.write(pause)
                total_frames += len(pause)
            output.write(audio)
            total_frames += len(audio)
    return total_frames


def _concat_file_line(path: str) -> str:
    # Escaping is ffconcat syntax, not shell quoting; subprocess receives argv.
    escaped = str(Path(path).resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


# ---------------------------------------------------------------------------
# Public: export_book
# ---------------------------------------------------------------------------


def export_book(
    chapters: List[Dict[str, Any]],
    output_dir: str,
    options: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, str]:
    """Stitch, normalize, and encode an audiobook in the requested format.

    Parameters
    ----------
    chapters:
        Ordered list of chapter dicts, each with keys:
          * ``number``        – int, 1-based chapter number
          * ``title``         – str, chapter display title
          * ``segment_paths`` – list of segment entries; each entry is either a
            plain path ``str`` **or** a ``dict`` with ``"path"`` (str) and
            optional ``"is_scene_break"`` (bool).  When ``is_scene_break`` is
            ``True``, the pause before that segment uses ``scene_pause_s``
            instead of ``inter_segment_pause_s``.

    output_dir:
        Directory where the output file will be written (created if absent).

    options:
        Dict controlling export behaviour:
          * ``format``        – ``"m4b"`` | ``"mp3_single"`` | ``"mp3_per_chapter"``
          * ``title``         – book title (for tags)
          * ``author``        – book author (for tags)
          * ``cover_path``    – optional path to cover image
          * ``bitrate``       – audio bitrate string, e.g. ``"64k"`` (default)
          * ``target_lufs``   – float, default −18.0
          * ``channels``      – int, 1 (mono) or 2 (stereo)
          * ``inter_segment_pause_s`` – float
          * ``scene_pause_s`` – float
          * ``chapter_pause_s`` – float
          * ``sample_rate``   – int

    progress_callback:
        Optional callable ``(percent: int, message: str) -> None`` invoked at
        key milestones so the caller (D7 SSE route) can stream progress.

    Returns
    -------
    (output_file_path, filename)
        Absolute path to the produced file, and the bare filename.
    """
    if options is None:
        options = {}

    fmt = options.get("format", "m4b")
    title = options.get("title", "Audiobook")
    author = options.get("author", "")
    cover_path = options.get("cover_path", None)
    bitrate = options.get("bitrate", DEFAULT_BITRATE)
    target_lufs = float(options.get("target_lufs", DEFAULT_TARGET_LUFS))
    channels = int(options.get("channels", DEFAULT_CHANNELS))
    sample_rate = int(options.get("sample_rate", DEFAULT_SAMPLE_RATE))
    inter_seg_pause = float(options.get("inter_segment_pause_s", DEFAULT_INTER_SEGMENT_PAUSE_S))
    scene_pause = float(options.get("scene_pause_s", DEFAULT_SCENE_PAUSE_S))
    chapter_pause = float(options.get("chapter_pause_s", DEFAULT_CHAPTER_PAUSE_S))

    def _progress(pct: int, msg: str = "") -> None:
        if progress_callback:
            progress_callback(pct, msg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _progress(0, "正在准备导出")

    if fmt not in {"m4b", "mp3_single", "mp3_per_chapter"}:
        raise ValueError(f"Unknown export format: {fmt!r}. Choose m4b, mp3_single, or mp3_per_chapter.")

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip() or "audiobook"
    # Temporary audio lives beside the export on the user's configured disk.
    # Closing this context removes staged chapters on both success and failure.
    with tempfile.TemporaryDirectory(prefix=".voiceit-export-", dir=output_dir) as work_dir:
        chapter_files: List[Tuple[Dict[str, Any], str]] = []
        chapter_starts_s: List[float] = []
        concat_lines = ["ffconcat version 1.0"]
        current_frames = 0
        pause_audio = _make_silence(chapter_pause, sample_rate)
        pause_path = str(Path(work_dir) / "chapter-pause.wav")
        sf.write(pause_path, pause_audio, sample_rate, subtype="PCM_16")
        n_chapters = len(chapters)

        for idx, chapter in enumerate(chapters):
            _progress(int(5 + idx / max(n_chapters, 1) * 50),
                      f"正在拼接第 {chapter.get('number', idx + 1)} 章")
            chapter_path = str(Path(work_dir) / f"chapter-{idx:06d}.wav")
            frames = _write_chapter_wav(chapter.get("segment_paths", []), chapter_path,
                                        sample_rate, inter_seg_pause, scene_pause)
            if not frames:
                continue
            chapter_starts_s.append(current_frames / sample_rate)
            chapter_files.append((chapter, chapter_path))
            concat_lines.append(_concat_file_line(chapter_path))
            current_frames += frames
            # Keep the original pause contract, including a pause after a valid
            # chapter followed by an entirely missing source chapter.
            if idx < n_chapters - 1:
                concat_lines.append(_concat_file_line(pause_path))
                current_frames += len(pause_audio)

        if not chapter_files:
            raise ValueError("No audio could be loaded from any segment in any chapter.")

        manifest = str(Path(work_dir) / "book.ffconcat")
        Path(manifest).write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        _progress(60, "正在准备响度标准化")

        if fmt == "m4b":
            out_path = str(Path(output_dir) / f"{safe_title}.m4b")
            staged_output = str(Path(work_dir) / "book.m4b")
            _progress(70, "正在编码 M4B 有声书")
            _encode_aac_m4b(
                manifest, staged_output, [chapter for chapter, _ in chapter_files], chapter_starts_s,
                title=title, author=author, cover_path=cover_path, bitrate=bitrate,
                channels=channels, concat=True, target_lufs=target_lufs,
                sample_rate=sample_rate, total_duration_s=current_frames / sample_rate,
            )
        elif fmt == "mp3_single":
            out_path = str(Path(output_dir) / f"{safe_title}.mp3")
            staged_output = str(Path(work_dir) / "book.mp3")
            _progress(70, "正在编码 MP3 有声书")
            _encode_mp3(manifest, staged_output, bitrate=bitrate, channels=channels,
                        concat=True, target_lufs=target_lufs, sample_rate=sample_rate)
            _embed_mp3_tags(staged_output, title=title, album=title, artist=author, cover_path=cover_path)
        else:
            out_path = str(Path(output_dir) / f"{safe_title}.zip")
            staged_output = str(Path(work_dir) / "book.zip")
            _progress(70, "正在逐章编码 MP3")
            with zipfile.ZipFile(staged_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for idx, (chapter, chapter_path) in enumerate(chapter_files):
                    number = chapter.get("number", idx + 1)
                    chapter_title = chapter.get("title", f"Chapter {number}")
                    _progress(int(70 + idx / len(chapter_files) * 25), f"正在编码《{chapter_title}》")
                    mp3_path = str(Path(work_dir) / f"chapter-{idx:06d}.mp3")
                    _encode_mp3(chapter_path, mp3_path, bitrate=bitrate, channels=channels,
                                target_lufs=target_lufs, sample_rate=sample_rate)
                    _embed_mp3_tags(mp3_path, title=chapter_title, album=title, artist=author,
                                    track_num=number, cover_path=cover_path)
                    chapter_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in chapter_title).strip()
                    archive.write(mp3_path, arcname=f"{number:02d}_{chapter_name}.mp3")
                    Path(mp3_path).unlink()

        # Publish only a fully encoded and tagged artifact. A failed re-export
        # cannot truncate an existing audiobook, and all staging shares its disk.
        os.replace(staged_output, out_path)

    _progress(100, "导出完成")
    return out_path, Path(out_path).name
