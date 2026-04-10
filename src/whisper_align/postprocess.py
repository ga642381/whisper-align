from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def detect_silence_gaps_fixed_threshold(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 20,
    hop_ms: int = 10,
    energy_threshold: float = 0.02,
    min_gap_duration: float = 0.2,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, float]:
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    energy = []
    times = []
    for idx in range(0, len(audio) - frame_length, hop_length):
        frame = audio[idx : idx + frame_length]
        rms = np.sqrt(np.mean(frame**2))
        energy.append(rms)
        times.append(idx / sr)

    energy_array = np.array(energy)
    time_array = np.array(times)
    is_silence = energy_array < energy_threshold

    gaps: list[dict[str, float]] = []
    in_gap = False
    gap_start = 0.0

    for timestamp, silent in zip(time_array, is_silence):
        if silent and not in_gap:
            gap_start = float(timestamp)
            in_gap = True
        elif not silent and in_gap:
            gap_duration = float(timestamp) - gap_start
            if gap_duration >= min_gap_duration:
                gaps.append(
                    {
                        "start": gap_start,
                        "end": float(timestamp),
                        "duration": gap_duration,
                    }
                )
            in_gap = False

    logger.debug(
        "Detected %d gaps with fixed threshold %.6f", len(gaps), energy_threshold
    )
    return gaps, energy_array, time_array, energy_threshold


def plot_energy_analysis(
    audio: np.ndarray,
    sr: int,
    words: list[dict[str, Any]],
    gaps: list[dict[str, float]],
    energy: np.ndarray,
    times: np.ndarray,
    threshold: float,
    save_path: str = "energy_analysis.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib is not installed; skipping energy plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    audio_times = np.arange(len(audio)) / sr
    ax1.plot(
        audio_times[::100],
        audio[::100],
        alpha=0.6,
        linewidth=0.5,
        color="steelblue",
    )
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.set_title(
        "Audio Waveform with Word Boundaries and Detected Gaps",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    for index, word in enumerate(words):
        ax1.axvline(word["start"], color="green", alpha=0.4, linestyle="--", linewidth=1)
        ax1.axvline(word["end"], color="red", alpha=0.4, linestyle="--", linewidth=1)
        y_pos = ax1.get_ylim()[1] * (0.95 - (index % 3) * 0.05)
        ax1.text(
            word["start"],
            y_pos,
            word["text"],
            rotation=45,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    for gap in gaps:
        ax1.axvspan(gap["start"], gap["end"], alpha=0.3, color="orange", label="_nolegend_")

    green_patch = mpatches.Patch(color="green", alpha=0.4, label="Word start")
    red_patch = mpatches.Patch(color="red", alpha=0.4, label="Word end")
    orange_patch = mpatches.Patch(color="orange", alpha=0.3, label="Detected gap")
    ax1.legend(handles=[green_patch, red_patch, orange_patch], loc="upper right", fontsize=10)

    ax2.plot(times, energy, label="Energy (RMS)", color="darkblue", linewidth=1.5)
    ax2.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {threshold:.6f}",
    )

    for gap in gaps:
        ax2.axvspan(gap["start"], gap["end"], alpha=0.3, color="orange", label="_nolegend_")
        mid_time = (gap["start"] + gap["end"]) / 2
        ax2.text(
            mid_time,
            ax2.get_ylim()[1] * 0.9,
            f"{gap['duration']:.2f}s",
            ha="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    for word in words:
        ax2.axvline(word["start"], color="green", alpha=0.2, linestyle="--", linewidth=0.5)
        ax2.axvline(word["end"], color="red", alpha=0.2, linestyle="--", linewidth=0.5)

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Energy (RMS)", fontsize=12)
    ax2.set_title("Audio Energy with Silence Detection", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    stats_text = (
        f"Duration: {len(audio) / sr:.2f}s\n"
        f"Words: {len(words)}\n"
        f"Gaps detected: {len(gaps)}\n"
        f"Energy min/max: {energy.min():.6f} / {energy.max():.6f}\n"
        f"Energy mean: {energy.mean():.6f}\n"
        f"Threshold: {threshold:.6f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved energy analysis plot to %s", save_path)


def correct_word_timestamps_with_gaps(
    words: list[dict[str, Any]],
    gaps: list[dict[str, float]],
) -> list[dict[str, Any]]:
    if not gaps or not words:
        logger.debug("No gaps or words; returning original timestamps")
        return words

    corrected_words = [word.copy() for word in words]
    adjustments = 0

    for gap in gaps:
        gap_start = gap["start"]
        gap_end = gap["end"]

        word_before_idx = None
        for index, word in enumerate(corrected_words):
            if word["end"] > gap_start and word["start"] < gap_start:
                word_before_idx = index
                break

        word_after_idx = None
        for index, word in enumerate(corrected_words):
            if word["start"] < gap_end and word["end"] > gap_end:
                word_after_idx = index
                break

        if (
            word_before_idx is not None
            and word_after_idx is not None
            and word_before_idx == word_after_idx
        ):
            word = corrected_words[word_before_idx]
            if (word["end"] - gap_end) > (gap_start - word["start"]):
                corrected_words[word_before_idx]["start"] = round(gap_end, 2)
            else:
                corrected_words[word_before_idx]["end"] = round(gap_start, 2)
            adjustments += 1
            continue

        if word_before_idx is not None:
            corrected_words[word_before_idx]["end"] = round(gap_start, 2)
            adjustments += 1

        if word_after_idx is not None:
            corrected_words[word_after_idx]["start"] = round(gap_end, 2)
            adjustments += 1

    logger.info("Made %d timestamp adjustments", adjustments)

    for index, word in enumerate(corrected_words):
        if word["start"] >= word["end"]:
            logger.warning(
                "Invalid timestamps for %r: start=%.2f >= end=%.2f; fixing",
                word["text"],
                word["start"],
                word["end"],
            )
            word["end"] = word["start"] + 0.1

        if index < len(corrected_words) - 1:
            next_word = corrected_words[index + 1]
            if word["end"] > next_word["start"]:
                logger.warning(
                    "Overlapping words: %r end=%.2f > %r start=%.2f; fixing",
                    word["text"],
                    word["end"],
                    next_word["text"],
                    next_word["start"],
                )
                midpoint = (word["end"] + next_word["start"]) / 2
                word["end"] = round(midpoint, 2)
                next_word["start"] = round(midpoint, 2)

    return corrected_words


def apply_energy_based_correction(
    pipe_output: dict[str, Any],
    audio: np.ndarray,
    sample_rate: int = 16000,
    energy_threshold: float = 0.02,
    min_gap_duration: float = 0.3,
    plot_path: str | None = None,
) -> dict[str, Any]:
    gaps, energy, times, threshold = detect_silence_gaps_fixed_threshold(
        audio,
        sr=sample_rate,
        energy_threshold=energy_threshold,
        min_gap_duration=min_gap_duration,
    )

    if not gaps:
        logger.warning("No silence gaps detected; returning original output")
        return pipe_output

    original_words: list[dict[str, Any]] = []
    for segment in pipe_output["segments"]:
        if "words" in segment:
            original_words.extend(segment["words"])

    if plot_path:
        plot_energy_analysis(
            audio,
            sample_rate,
            original_words,
            gaps,
            energy,
            times,
            threshold,
            save_path=plot_path.replace(".png", "_before.png"),
        )

    corrected_output = pipe_output.copy()
    corrected_output["segments"] = []

    for segment in pipe_output["segments"]:
        corrected_segment = segment.copy()
        if "words" in segment and segment["words"]:
            corrected_segment["words"] = correct_word_timestamps_with_gaps(
                segment["words"], gaps
            )
            if corrected_segment["words"]:
                corrected_segment["start"] = corrected_segment["words"][0]["start"]
                corrected_segment["end"] = corrected_segment["words"][-1]["end"]
        corrected_output["segments"].append(corrected_segment)

    corrected_words: list[dict[str, Any]] = []
    for segment in corrected_output["segments"]:
        if "words" in segment:
            corrected_words.extend(segment["words"])

    if plot_path:
        plot_energy_analysis(
            audio,
            sample_rate,
            corrected_words,
            gaps,
            energy,
            times,
            threshold,
            save_path=plot_path.replace(".png", "_after.png"),
        )

    logger.info("Applied energy-based timestamp correction")
    return corrected_output
