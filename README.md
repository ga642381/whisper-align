# whisper-align

`whisper-align` packages the Whisper word-alignment workflow that is currently used internally by the Game-Time and TiCo projects.

This code was extracted from local annotation scripts and is explicitly maintained as a modified derivative of the `moshi-finetune` repo. The goal of this repo is practical internal reuse: one installable alignment package that can run locally on a single GPU, run a single shard, or fan out shards through SLURM with cluster-specific settings moved into config.

## Status

- Intended for internal usage in Game-Time and TiCo.
- Source of truth for the shared Whisper alignment pipeline.
- Cluster-specific SLURM details belong in YAML, not in the core alignment code.

## Install

From GitHub into another project with `uv add`:

```bash
uv add "whisper-align @ git+https://github.com/ga642381/whisper-align.git"
```

With SLURM support:

```bash
uv add "whisper-align[slurm] @ git+https://github.com/ga642381/whisper-align.git"
```

If you are working inside the package repo itself:

```bash
cd whisper-align
uv pip install -e .
```

Editable install with SLURM support:

```bash
cd whisper-align
uv pip install -e ".[slurm]"
```

The package also installs a CLI:

```bash
whisper-align --help
```

## Local Usage

```bash
whisper-align \
  --in-dir /path/to/audio \
  --out-dir /path/to/alignment \
  --lang en \
  --whisper-model medium \
  --channel-mode single \
  --keep-silence-in-segments \
  --local
```

## SLURM Usage

Using a built-in SLURM profile bundled with the package:

```bash
whisper-align \
  --in-dir /path/to/audio \
  --out-dir /path/to/alignment \
  --lang en \
  --whisper-model medium \
  --channel-mode single \
  --keep-silence-in-segments \
  --shards 4 \
  --slurm-profile a5
```

You can also pass your own config file:

```bash
whisper-align \
  --in-dir /path/to/audio \
  --out-dir /path/to/alignment \
  --shards 4 \
  --slurm-config /path/to/slurm.yaml
```

Add a new YAML file or packaged profile for each cluster or queue policy instead of hardcoding machine-specific parameters in Python.

## Python API

Downstream projects should import the package API directly rather than calling `cli.main()`:

```python
from pathlib import Path

from whisper_align import AlignParams, resolve_slurm_config, run, submit_and_monitor

params = AlignParams(
    in_dir=Path("/path/to/audio"),
    out_dir=Path("/path/to/alignment"),
    whisper_model="medium",
    channel_mode="single",
    keep_silence_in_segments=1.0,
    rerun_errors=True,
)

run(params)

slurm = resolve_slurm_config(
    log_folder=Path("/tmp/whisper-align"),
    profile_name="a5",
)
submit_and_monitor(params, slurm)
```

## Notes

- Output mirrors the input directory structure and writes `.jsonl`.
- Existing `.jsonl` outputs are skipped automatically.
- Failed files get a `.json.err` marker unless the error is CUDA-related.
- Energy-based postprocessing for word timestamps is enabled by default and can be disabled from the CLI.
