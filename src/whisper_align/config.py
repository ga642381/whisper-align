from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ChannelMode = Literal["single", "dual"]


@dataclass(slots=True)
class AlignParams:
    in_dir: Path
    out_dir: Path
    lang: str = "en"
    channel_mode: ChannelMode = "single"
    whisper_model: str = "medium"
    verbose: bool = False
    keep_silence_in_segments: float = 0.0
    correct_word_timestamps_with_gaps: bool = True
    rerun_errors: bool = False
    shards: int = 1
    shard: int = 0
    device: str = "cuda"
    min_file_size_bytes: int = 1000
    max_duration_seconds: float = 4 * 3600
    silence_energy_threshold: float = 1e-8
    gap_energy_threshold: float = 0.02
    min_gap_duration: float = 0.3
    single_channel_speaker: str = "SPEAKER_USER"
    dual_channel_speaker_0: str = "SPEAKER_AGENT"
    dual_channel_speaker_1: str = "SPEAKER_USER"
    omp_num_threads: int = 2


@dataclass(slots=True)
class SlurmConfig:
    folder: Path
    partition: str = ""
    qos: str = "regular"
    cpus_per_task: int = 6
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    time_minutes: int = 60 * 24
    signal_delay_s: int = 30
    stderr_to_stdout: bool = True
    array_parallelism: int = 1000
    exclude: str = ""
    job_name: str = "annotate"
    mem: str = "64G"
    additional_parameters: dict[str, Any] = field(default_factory=dict)
