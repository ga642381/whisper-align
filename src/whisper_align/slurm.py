from __future__ import annotations

from importlib.resources import as_file, files
import time
from pathlib import Path

from whisper_align.config import AlignParams, SlurmConfig


def load_slurm_config(
    config_path: Path | None,
    log_folder: Path,
    partition_override: str = "",
) -> SlurmConfig:
    data: dict = {"folder": log_folder}

    if config_path is not None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for --slurm-config. Install whisper-align with the [slurm] extra."
            ) from exc

        raw = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"SLURM config {config_path} must contain a mapping at the top level.")
        data.update(raw)

    if "time" in data and "time_minutes" not in data:
        data["time_minutes"] = data.pop("time")

    data["folder"] = Path(data.get("folder", log_folder)).expanduser()
    data.setdefault("additional_parameters", {})
    config = SlurmConfig(**data)

    if partition_override:
        config.partition = partition_override

    return config


def available_slurm_profiles() -> list[str]:
    profile_dir = files("whisper_align.configs")
    profiles = []
    for path in profile_dir.iterdir():
        if path.suffix != ".yaml":
            continue
        name = path.stem
        if name.startswith("slurm_"):
            name = name.removeprefix("slurm_")
        profiles.append(name)
    return sorted(set(profiles))


def load_slurm_profile(
    profile_name: str,
    log_folder: Path,
    partition_override: str = "",
) -> SlurmConfig:
    profile_dir = files("whisper_align.configs")
    candidates = [
        profile_dir.joinpath(f"{profile_name}.yaml"),
        profile_dir.joinpath(f"slurm_{profile_name}.yaml"),
    ]
    profile_path = next((path for path in candidates if path.is_file()), None)
    if profile_path is None:
        known = ", ".join(available_slurm_profiles()) or "<none>"
        raise ValueError(f"Unknown SLURM profile {profile_name!r}. Available profiles: {known}")
    with as_file(profile_path) as resolved_path:
        return load_slurm_config(resolved_path, log_folder, partition_override)


def resolve_slurm_config(
    *,
    log_folder: Path,
    config_path: Path | None = None,
    profile_name: str | None = None,
    partition_override: str = "",
) -> SlurmConfig:
    if config_path is not None and profile_name is not None:
        raise ValueError("Pass either config_path or profile_name, not both.")
    if profile_name is not None:
        return load_slurm_profile(profile_name, log_folder, partition_override)
    return load_slurm_config(config_path, log_folder, partition_override)


def make_executor(config: SlurmConfig):
    try:
        import submitit
    except ImportError as exc:
        raise RuntimeError(
            "submitit is required for SLURM runs. Install whisper-align with the [slurm] extra."
        ) from exc

    executor = submitit.SlurmExecutor(folder=config.folder)
    additional_parameters = dict(config.additional_parameters)
    if config.mem:
        additional_parameters.setdefault("mem", config.mem)

    kwargs = {
        "cpus_per_task": config.cpus_per_task,
        "ntasks_per_node": config.ntasks_per_node,
        "gpus_per_node": config.gpus_per_node,
        "time": config.time_minutes,
        "signal_delay_s": config.signal_delay_s,
        "stderr_to_stdout": config.stderr_to_stdout,
        "array_parallelism": config.array_parallelism,
        "job_name": config.job_name,
        "additional_parameters": additional_parameters,
    }
    if config.partition:
        kwargs["partition"] = config.partition
    if config.qos:
        kwargs["qos"] = config.qos
    if config.exclude:
        kwargs["exclude"] = config.exclude

    executor.update_parameters(**kwargs)
    return executor


def submit_shards(params: AlignParams, config: SlurmConfig):
    from whisper_align.runner import run

    executor = make_executor(config)
    jobs = []
    with executor.batch():
        for shard in range(params.shards):
            jobs.append(executor.submit(run, params, shard))
    return jobs


def submit_and_monitor(params: AlignParams, config: SlurmConfig):
    jobs = submit_shards(params, config)
    monitor_jobs(jobs)
    return jobs


def monitor_jobs(jobs, poll_seconds: float = 10.0) -> None:
    if not jobs:
        return
    print("Job id:", jobs[0].job_id)
    while True:
        done = sum(job.done() for job in jobs)
        print(f"{done:04d} / {len(jobs):04d} jobs done.", end="\r")
        if done == len(jobs):
            print()
            return
        time.sleep(poll_seconds)
