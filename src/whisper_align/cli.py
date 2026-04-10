from __future__ import annotations

import argparse
import logging
from pathlib import Path

from whisper_align.config import AlignParams
from whisper_align.core import init_logging
from whisper_align.runner import run
from whisper_align.slurm import available_slurm_profiles, resolve_slurm_config, submit_and_monitor

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whisper word alignment for local and SLURM execution."
    )
    parser.add_argument(
        "--in-dir",
        "--in_dir",
        dest="in_dir",
        type=Path,
        required=True,
        help="Input directory containing .wav files.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for alignment files.",
    )
    parser.add_argument(
        "--log-folder",
        "--log_folder",
        dest="log_folder",
        type=Path,
        default=Path.home() / "tmp" / "whisper-align",
        help="Folder used by submitit for SLURM logs.",
    )
    parser.add_argument(
        "-S",
        "--shards",
        type=int,
        default=1,
        help="Number of shards to schedule.",
    )
    parser.add_argument(
        "--shard",
        "--shard-index",
        dest="shard",
        type=int,
        default=0,
        help="Shard index to run locally.",
    )
    parser.add_argument("--lang", default="en", help="Force the language.")
    parser.add_argument(
        "--partition",
        default="",
        help="Optional SLURM partition override.",
    )
    parser.add_argument(
        "--slurm-config",
        type=Path,
        default=None,
        help="Optional YAML file with cluster-specific submitit parameters.",
    )
    parser.add_argument(
        "--slurm-profile",
        choices=available_slurm_profiles(),
        default=None,
        help="Optional built-in SLURM profile shipped with the package.",
    )
    parser.add_argument(
        "--channel-mode",
        "--channel_mode",
        dest="channel_mode",
        choices=["single", "dual"],
        default="single",
        help="single = channel 0 only, dual = channel 0 plus channel 1.",
    )
    parser.add_argument(
        "--whisper-model",
        "--whisper_model",
        dest="whisper_model",
        default="medium",
        help="Whisper model name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Runtime device for local execution: cuda, cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--rerun-errors",
        "--rerun_errors",
        dest="rerun_errors",
        action="store_true",
        help="Rerun files that already have a .json.err marker.",
    )
    parser.add_argument(
        "--keep-silence-in-segments",
        "--keep_silence_in_segments",
        dest="keep_silence_in_segments",
        nargs="?",
        const=1.0,
        default=0.0,
        type=float,
        help="Reintroduce up to N seconds of boundary silence into VAD segments. Passing the flag alone uses 1.0 second.",
    )
    parser.add_argument(
        "--correct-word-timestamps-with-gaps",
        dest="correct_word_timestamps_with_gaps",
        action="store_true",
        default=True,
        help="Apply energy-based correction to word timestamps.",
    )
    parser.add_argument(
        "--no-correct-word-timestamps-with-gaps",
        dest="correct_word_timestamps_with_gaps",
        action="store_false",
        help="Disable energy-based correction to word timestamps.",
    )
    parser.add_argument(
        "--gap-energy-threshold",
        type=float,
        default=0.02,
        help="Fixed RMS energy threshold for silence-gap detection.",
    )
    parser.add_argument(
        "--min-gap-duration",
        type=float,
        default=0.3,
        help="Minimum silence duration to count as a removable gap.",
    )
    parser.add_argument(
        "--silence-energy-threshold",
        type=float,
        default=1e-8,
        help="Early-exit threshold for nearly silent audio.",
    )
    parser.add_argument(
        "--min-file-size-bytes",
        type=int,
        default=1000,
        help="Skip tiny files below this size.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=4 * 3600,
        help="Reject audio files longer than this.",
    )
    parser.add_argument("-l", "--local", action="store_true", help="Run locally.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def _build_params(args: argparse.Namespace) -> AlignParams:
    return AlignParams(
        in_dir=args.in_dir.expanduser(),
        out_dir=args.out_dir.expanduser(),
        lang=args.lang,
        channel_mode=args.channel_mode,
        whisper_model=args.whisper_model,
        verbose=args.verbose,
        keep_silence_in_segments=args.keep_silence_in_segments,
        correct_word_timestamps_with_gaps=args.correct_word_timestamps_with_gaps,
        rerun_errors=args.rerun_errors,
        shards=args.shards,
        shard=args.shard,
        device=args.device,
        min_file_size_bytes=args.min_file_size_bytes,
        max_duration_seconds=args.max_duration_seconds,
        silence_energy_threshold=args.silence_energy_threshold,
        gap_energy_threshold=args.gap_energy_threshold,
        min_gap_duration=args.min_gap_duration,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    init_logging(args.verbose)

    if args.shards < 1:
        parser.error("--shards must be >= 1")
    if args.shard < 0:
        parser.error("--shard must be >= 0")
    if args.local and args.shard >= args.shards:
        parser.error("--shard must be smaller than --shards for local runs")

    if args.whisper_model == "large-v3":
        logger.warning("medium is usually the safer default for stereo plus VAD.")

    params = _build_params(args)

    if args.local:
        run(params, shard=args.shard)
        return 0

    if args.shard != 0:
        logger.warning("--shard is ignored for SLURM submission; all shards will be scheduled.")

    slurm_config = resolve_slurm_config(
        log_folder=args.log_folder.expanduser(),
        config_path=args.slurm_config,
        profile_name=args.slurm_profile,
        partition_override=args.partition,
    )
    submit_and_monitor(params, slurm_config)
    return 0
