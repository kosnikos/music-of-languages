"""Capture adapters: turn one RecordingRef into one 16 kHz mono wav.

Throwaway-quality probes. ffmpeg/streamlink are invoked via an injectable
`runner` so unit tests never shell out (mirrors ingest.radio.record_clip).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import requests

from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.ingest.radio import record_clip
from musiclang.probe.core import RecordingRef

USER_AGENT = "music-of-languages/0.0 (research; contact: project owner)"


def _seconds_to_hhmmss(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def capture_progressive(
    ref: RecordingRef, out_path, *, duration_s: int = 60, runner=subprocess.run
) -> Path | None:
    """Capture a progressive (icecast/shoutcast) stream via ffmpeg (reuses record_clip)."""
    try:
        return record_clip(ref.ref, out_path, duration_s=duration_s, runner=runner)
    except Exception:
        return None


def capture_hls(
    ref: RecordingRef, out_path, *, duration_s: int = 60, runner=subprocess.run
) -> Path | None:
    """Capture an HLS (.m3u8) slice with streamlink, then transcode to 16k mono wav."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "hls.ts"
        streamlink_cmd = [
            "streamlink", "--hls-live-restart",
            "--hls-duration", _seconds_to_hhmmss(duration_s),
            ref.ref, "best", "-o", str(media),
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(media),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), str(out_path),
        ]
        try:
            runner(streamlink_cmd, check=True)
            if not media.exists():
                return None
            runner(ffmpeg_cmd, check=True)
        except Exception:
            return None
    return out_path if out_path.exists() else None


def _download(url: str, dest: Path) -> Path | None:
    try:
        with requests.get(
            url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}
        ) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        return dest
    except Exception:
        return None


def capture_rss(
    ref: RecordingRef, out_path, *, skip_s: int = 30, take_s: int = 90,
    downloader=None, runner=subprocess.run,
) -> Path | None:
    """Download an episode enclosure, transcode the [skip_s, skip_s+take_s] slice to wav.

    The skip skips a likely intro jingle so the measured slice is mostly talk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = downloader or _download
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "episode.media"
        if downloader(ref.ref, media) is None:
            return None
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(skip_s), "-t", str(take_s), "-i", str(media),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), str(out_path),
        ]
        try:
            runner(ffmpeg_cmd, check=True)
        except Exception:
            return None
    return out_path if out_path.exists() else None


def capture_local(ref: RecordingRef, out_path=None, **_) -> Path | None:
    """Corpus refs already point at a written wav — just hand it back if present."""
    p = Path(ref.ref)
    return p if p.exists() else None


def latest_enclosures(feed_url: str, k: int, *, parser=None) -> list[str]:
    """Up to k audio enclosure URLs from a podcast RSS feed (feed order, newest first)."""
    if parser is None:
        import feedparser
        parser = feedparser.parse
    feed = parser(feed_url)
    urls: list[str] = []
    for entry in getattr(feed, "entries", [])[: k * 3]:
        for enc in getattr(entry, "enclosures", []) or []:
            href = enc.get("href") or enc.get("url")
            if href and "audio" in (enc.get("type", "") or ""):
                urls.append(href)
                break
        if len(urls) >= k:
            break
    return urls[:k]


CAPTURE_DISPATCH = {
    "progressive": capture_progressive,
    "hls": capture_hls,
    "rss": capture_rss,
    "corpus": capture_local,
}


# ---------------------------------------------------------------------------
# Corpus probe: VoxPopuli (7 languages) + Common Voice 16.0 (Greek)
# ---------------------------------------------------------------------------

# Maps language -> (dataset_id, config, split, speaker_field)
# Greek has no transcribed VoxPopuli split; Common Voice 16.0 provides it.
_CORPUS_SPEC: dict[str, tuple[str, str, str, str]] = {
    "english": ("facebook/voxpopuli", "en", "test", "speaker_id"),
    "german":  ("facebook/voxpopuli", "de", "test", "speaker_id"),
    "polish":  ("facebook/voxpopuli", "pl", "test", "speaker_id"),
    "french":  ("facebook/voxpopuli", "fr", "test", "speaker_id"),
    "spanish": ("facebook/voxpopuli", "es", "test", "speaker_id"),
    "italian": ("facebook/voxpopuli", "it", "test", "speaker_id"),
    "finnish": ("facebook/voxpopuli", "fi", "test", "speaker_id"),
    "greek":   ("mozilla-foundation/common_voice_16_0", "el", "test", "client_id"),
}


def _load_corpus(language: str):
    """Streaming corpus dataset for `language`.

    VoxPopuli (Wang et al. 2021, arXiv:2101.00390) for 7 languages;
    Common Voice 16.0 (mozilla-foundation) for Greek which has no VoxPopuli split.
    Common Voice is gated — the caller is expected to handle auth failures gracefully.
    """
    from datasets import load_dataset

    dataset_id, config, split, _ = _CORPUS_SPEC[language]
    kwargs: dict = {"split": split, "streaming": True}
    if dataset_id.startswith("mozilla-foundation/"):
        # Common Voice uses a loading script and is a gated dataset.
        kwargs["trust_remote_code"] = True
    return load_dataset(dataset_id, config, **kwargs)


def corpus_probe(
    language: str, n: int, out_dir, *, loader=None, writer=None
) -> "list[tuple[RecordingRef, Path]]":
    """Pull up to `n` distinct-speaker corpus samples; write 16 kHz mono wavs.

    Deduplication by speaker id (VoxPopuli: ``speaker_id``; Common Voice: ``client_id``)
    ensures each returned recording comes from a different speaker, giving independent
    samples for the prosody probe.

    Parameters
    ----------
    language:
        One of the eight SEED_LANGUAGES keys.
    n:
        Maximum number of distinct-speaker recordings to return.
    out_dir:
        Directory in which wavs are written (created if absent).
    loader:
        Callable ``(language) -> iterable``; defaults to ``_load_corpus``.
        Injectable for tests so no network access is required.
    writer:
        Callable ``(path, array, sample_rate)``; defaults to ``soundfile.write``.
        Injectable for tests.

    Returns
    -------
    list of ``(RecordingRef, Path)`` pairs, at most `n` entries.
    """
    import soundfile as sf

    loader = loader or _load_corpus
    writer = writer or sf.write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_field = _CORPUS_SPEC[language][3]

    pairs: list[tuple[RecordingRef, Path]] = []
    seen: set[str] = set()
    for item in loader(language):
        # Robustly extract speaker id across both dataset shapes.
        spk = str(
            item.get(speaker_field)
            or item.get("speaker_id")
            or item.get("client_id")
            or f"spk{len(pairs)}"
        )
        if spk in seen:
            continue
        seen.add(spk)
        audio = item["audio"]
        wav = out_dir / f"corpus_{language}_{len(pairs):02d}.wav"
        writer(str(wav), audio["array"], audio["sampling_rate"])
        ref = RecordingRef("corpus", language, spk, "corpus", str(wav))
        pairs.append((ref, wav))
        if len(pairs) >= n:
            break
    return pairs
