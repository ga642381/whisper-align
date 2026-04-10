from __future__ import annotations

import gc
import importlib
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sphn
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper

from whisper_align.config import AlignParams
from whisper_align.postprocess import apply_energy_based_correction

transcribe = importlib.import_module("whisper_timestamped.transcribe")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@contextmanager
def write_and_rename(
    path: Path,
    mode: str = "wb",
    suffix: str = ".tmp",
    pid: bool = False,
) -> Iterator[object]:
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as handle:
        yield handle
    os.rename(tmp_path, path)


def init_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


def load_audio_paths(in_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for extension in ("*.wav", "*.WAV"):
        paths.extend(in_dir.rglob(extension))
    return sorted(paths)


def get_output_path(in_path: Path, in_dir: Path, out_dir: Path) -> Path:
    return out_dir / in_path.relative_to(in_dir).with_suffix(".jsonl")


def resolve_runtime_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    return device


@contextmanager
def patched_vad_boundaries(keep_silence_in_segments: float) -> Iterator[None]:
    if keep_silence_in_segments <= 0:
        yield
        return

    original_get_vad_segments = transcribe.get_vad_segments

    def new_get_vad_segments(*args, **kwargs):
        segments = original_get_vad_segments(*args, **kwargs)
        padded_segments = []
        last_end = 0
        pad = int(SAMPLE_RATE * keep_silence_in_segments)
        logger.debug("Reintroducing %d samples at segment boundaries", pad)
        for segment in segments:
            padded_segments.append(
                {
                    "start": max(last_end, segment["start"] - pad),
                    "end": segment["end"] + pad,
                }
            )
            last_end = padded_segments[-1]["end"]
        return padded_segments

    transcribe.get_vad_segments = new_get_vad_segments
    try:
        yield
    finally:
        transcribe.get_vad_segments = original_get_vad_segments


def process_one(
    in_file: Path,
    language: str,
    w_model,
    params: AlignParams,
    channel: int = 0,
    speaker: str = "SPEAKER_MAIN",
    seek_time: float | None = None,
    duration: float | None = None,
    runtime_device: torch.device | None = None,
) -> dict[str, list[list[object]]]:
    logger.debug("Loading audio %s", in_file)
    runtime_device = runtime_device or resolve_runtime_device(params.device)

    gc.collect()
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()

    audio, sample_rate = sphn.read(in_file, start_sec=seek_time, duration_sec=duration)
    waveform = torch.from_numpy(audio).to(runtime_device)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    duration_seconds = waveform.shape[-1] / sample_rate
    if duration_seconds > params.max_duration_seconds:
        raise RuntimeError(
            f"File is too long ({duration_seconds:.1f}s > {params.max_duration_seconds:.1f}s)."
        )

    if channel >= waveform.shape[0]:
        raise RuntimeError(
            f"Requested channel {channel} but {in_file} only has {waveform.shape[0]} channel(s)."
        )

    vocals = F.resample(waveform[channel][None], sample_rate, SAMPLE_RATE)
    vocals_np = vocals.detach().cpu().numpy()[0]
    energy = float((vocals_np**2).mean())
    if energy < params.silence_energy_threshold:
        return {"alignments": []}

    logger.debug("Transcribing %.1fs block in %s", vocals_np.shape[-1] / SAMPLE_RATE, language)

    with patched_vad_boundaries(params.keep_silence_in_segments):
        pipe_output = whisper.transcribe(
            w_model,
            vocals_np,
            language=language,
            vad="auditok",
            best_of=5,
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            verbose=None,
        )

    if params.correct_word_timestamps_with_gaps:
        pipe_output = apply_energy_based_correction(
            pipe_output,
            vocals_np,
            SAMPLE_RATE,
            energy_threshold=params.gap_energy_threshold,
            min_gap_duration=params.min_gap_duration,
        )

    chunks = []
    for segment in pipe_output["segments"]:
        if "words" not in segment:
            logger.error("No words in %s: %r", in_file, segment)
            continue
        for word in segment["words"]:
            if "text" not in word or "start" not in word or "end" not in word:
                logger.error("Missing key in %s: %r", in_file, word)
                raise KeyError(f"Missing word timing data in {in_file}")
            chunks.append([word["text"], [word["start"], word["end"]], speaker])

    logger.debug("Whisper applied to %s", in_file)
    return {"alignments": chunks}
