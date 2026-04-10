from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import whisper_timestamped as whisper

from whisper_align.config import AlignParams
from whisper_align.core import (
    get_output_path,
    init_logging,
    load_audio_paths,
    process_one,
    resolve_runtime_device,
    write_and_rename,
)

logger = logging.getLogger(__name__)


def _write_single_output(out_file: Path, output: dict) -> None:
    with write_and_rename(out_file, "w", pid=True) as handle:
        json.dump(output, handle, ensure_ascii=False)


def _write_dual_output(out_file: Path, first: dict, second: dict) -> None:
    with write_and_rename(out_file, "w", pid=True) as handle:
        for output in (first, second):
            json.dump(output, handle, ensure_ascii=False)
            handle.write("\n")


def run(params: AlignParams, shard: int | None = None) -> None:
    init_logging(params.verbose)
    if shard is not None:
        params.shard = shard

    if params.shards < 1:
        raise ValueError("shards must be >= 1")
    if params.shard < 0 or params.shard >= params.shards:
        raise ValueError(f"shard index {params.shard} must be in [0, {params.shards})")

    logger.info("Starting shard %d / %d", params.shard, params.shards)
    os.environ.setdefault("OMP_NUM_THREADS", str(params.omp_num_threads))

    runtime_device = resolve_runtime_device(params.device)
    if runtime_device.type == "cuda":
        import torch

        device_index = runtime_device.index if runtime_device.index is not None else 0
        torch.cuda.set_device(device_index)

    logger.info("Loading Whisper model %s on %s", params.whisper_model, runtime_device)
    w_model = whisper.load_model(params.whisper_model, device=runtime_device)

    logger.info("Scanning audio files under %s", params.in_dir)
    paths = load_audio_paths(params.in_dir)
    pending_paths = [
        path
        for path in paths
        if not get_output_path(path, params.in_dir, params.out_dir).exists()
    ]

    kept_paths = pending_paths[params.shard :: params.shards]
    logger.info("Processing %8d files out of %8d pending", len(kept_paths), len(pending_paths))

    for index, path in enumerate(kept_paths, start=1):
        if index % 100 == 0:
            logger.info("Processed %8d / %8d files in this shard", index, len(kept_paths))

        out_file = get_output_path(path, params.in_dir, params.out_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        err_file = out_file.with_suffix(".json.err")

        if out_file.exists():
            continue
        if err_file.exists() and not params.rerun_errors:
            continue

        try:
            if path.stat().st_size < params.min_file_size_bytes:
                logger.warning("Skipping small file %s", path)
                continue

            if params.channel_mode == "single":
                output = process_one(
                    path,
                    language=params.lang,
                    w_model=w_model,
                    params=params,
                    channel=0,
                    speaker=params.single_channel_speaker,
                    runtime_device=runtime_device,
                )
                _write_single_output(out_file, output)
            else:
                first = process_one(
                    path,
                    language=params.lang,
                    w_model=w_model,
                    params=params,
                    channel=0,
                    speaker=params.dual_channel_speaker_0,
                    runtime_device=runtime_device,
                )
                second = process_one(
                    path,
                    language=params.lang,
                    w_model=w_model,
                    params=params,
                    channel=1,
                    speaker=params.dual_channel_speaker_1,
                    runtime_device=runtime_device,
                )
                _write_dual_output(out_file, first, second)
            if err_file.exists():
                err_file.unlink()
        except Exception as err:
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
