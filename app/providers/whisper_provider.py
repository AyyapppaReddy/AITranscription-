"""
WhisperProvider – high-performance parallel transcription using faster-whisper.

Uses:
  - Configurable model size with int8 compute type on CPU
  - Parallel multi-worker chunk processing across CPU cores
  - Fast mode that processes only the first 5-minute window for a quick preview
  - Optional word-level timestamps (disabled by default for 2-3x speedup)
  - Configurable beam size (1 for ultra-fast greedy decoding)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.providers.base import (
    SegmentResult,
    TranscriptionProvider,
    TranscriptionResult,
    WordResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    _WhisperModel = None

class WhisperProvider(TranscriptionProvider):
    """
    faster-whisper backend with parallel multi-worker chunk processing.
    """

    def __init__(self) -> None:
        self._model = None
        self._loaded_model_size: Optional[str] = None
        self._num_workers: int = 1

    @property
    def name(self) -> str:
        return "whisper"

    def _ensure_model(self) -> None:
        model_size = settings.WHISPER_MODEL_SIZE
        if self._model is not None and self._loaded_model_size == model_size:
            return

        if not _FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )

        total_cores = settings.WHISPER_CPU_THREADS if settings.WHISPER_CPU_THREADS > 0 else (os.cpu_count() or 4)
        num_workers = 4 if settings.WHISPER_DEVICE == "cpu" else 1
        per_worker_threads = max(1, total_cores // num_workers)

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s, workers=%d, threads_per_worker=%d)",
            model_size,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
            num_workers,
            per_worker_threads,
        )
        self._model = _WhisperModel(
            model_size,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            cpu_threads=per_worker_threads,
            num_workers=num_workers,
        )
        self._loaded_model_size = model_size
        self._num_workers = num_workers
        logger.info("Whisper model initialized with %d parallel workers.", num_workers)

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        import wave
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception:
            pass
        return None

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "auto",
        vocabulary_hint: Optional[str] = None,
        fast_mode: bool = False,
    ) -> TranscriptionResult:
        self._ensure_model()

        lang_arg = None if language == "auto" else language
        duration = self._get_audio_duration(audio_path)
        logger.info("Transcribe called on '%s' (duration=%.1fs, fast_mode=%s)", audio_path.name, duration or 0, fast_mode)

        target_duration = duration
        if fast_mode and duration and duration > 300.0:
            target_duration = 300.0
            logger.info("Fast mode active: processing first 300s window")

        if target_duration and target_duration > 60.0 and self._num_workers > 1:
            raw_segments, detected_lang = self._transcribe_parallel(
                audio_path,
                target_duration=target_duration,
                language=lang_arg,
                vocabulary_hint=vocabulary_hint,
                num_workers=self._num_workers,
            )
        else:
            raw_segments, detected_lang = self._transcribe_single(
                audio_path,
                language=lang_arg,
                vocabulary_hint=vocabulary_hint,
                clip_str=f"0,{target_duration:.1f}" if (target_duration and duration and target_duration < duration) else None,
            )

        results: list[SegmentResult] = []
        for idx, seg in enumerate(raw_segments):
            words: list[WordResult] = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    words.append(
                        WordResult(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

            confidence: Optional[float] = None
            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                import math
                confidence = round(math.exp(seg.avg_logprob), 4)

            results.append(
                SegmentResult(
                    segment_index=idx,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=confidence,
                    words=words,
                )
            )

        logger.info(
            "Transcription complete: %d segments, detected_lang=%s",
            len(results),
            detected_lang,
        )

        return TranscriptionResult(
            segments=results,
            language_detected=detected_lang,
            provider_name=self.name,
            model_name=settings.WHISPER_MODEL_SIZE,
        )

    def _transcribe_parallel(
        self,
        audio_path: Path,
        *,
        target_duration: float,
        language: Optional[str],
        vocabulary_hint: Optional[str],
        num_workers: int,
    ) -> tuple[list, Optional[str]]:
        from concurrent.futures import ThreadPoolExecutor

        chunk_sec = target_duration / num_workers
        clips = [f"{i * chunk_sec:.1f},{(i + 1) * chunk_sec:.1f}" for i in range(num_workers)]

        def _worker(clip_str: str):
            segs, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=settings.WHISPER_BEAM_SIZE,
                word_timestamps=settings.WHISPER_WORD_TIMESTAMPS,
                condition_on_previous_text=False,
                initial_prompt=vocabulary_hint or None,
                clip_timestamps=clip_str,
            )
            return list(segs), getattr(info, "language", None)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_worker, clips))

        all_segs = []
        detected_lang = None
        for seg_list, lang in results:
            all_segs.extend(seg_list)
            if not detected_lang and lang:
                detected_lang = lang

        all_segs.sort(key=lambda s: s.start)
        return all_segs, detected_lang

    def _transcribe_single(
        self,
        audio_path: Path,
        *,
        language: Optional[str],
        vocabulary_hint: Optional[str],
        clip_str: Optional[str] = None,
    ) -> tuple[list, Optional[str]]:
        kwargs = {
            "language": language,
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "vad_filter": True,
            "word_timestamps": settings.WHISPER_WORD_TIMESTAMPS,
            "condition_on_previous_text": False,
            "initial_prompt": vocabulary_hint or None,
        }
        if clip_str:
            kwargs["clip_timestamps"] = clip_str
            kwargs["vad_filter"] = False

        segs_gen, info = self._model.transcribe(str(audio_path), **kwargs)
        return list(segs_gen), getattr(info, "language", None)
