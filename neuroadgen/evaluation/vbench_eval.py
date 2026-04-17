"""
VBench video quality evaluation integration.

VBench (https://github.com/Vchitect/VBench) measures 16 video generation
quality dimensions including subject consistency, motion smoothness,
aesthetic quality, and temporal coherence.

This module wraps VBench and provides a simplified scoring interface
for evaluating NeuroAdGen output videos alongside brain scores.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# VBench dimensions most relevant to ad quality
AD_QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def run_vbench(
    video_path: str,
    output_dir: str,
    dimensions: Optional[list[str]] = None,
    vbench_path: Optional[str] = None,
) -> dict:
    """
    Run VBench evaluation on a video file.

    Parameters
    ----------
    video_path  : Path to the video to evaluate.
    output_dir  : Directory to write VBench results.
    dimensions  : List of VBench dimensions (defaults to AD_QUALITY_DIMENSIONS).
    vbench_path : Path to VBench repo if not on PYTHONPATH.

    Returns
    -------
    dict mapping dimension name → score (float in [0, 1]).
    """
    dims = dimensions or AD_QUALITY_DIMENSIONS
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Prefer Python API if available
        return _vbench_python_api(video_path, output_dir, dims, vbench_path)
    except ImportError:
        logger.info("VBench Python API not found — falling back to CLI subprocess.")
        return _vbench_cli(video_path, output_dir, dims, vbench_path)


def _vbench_python_api(
    video_path: str,
    output_dir: str,
    dims: list[str],
    vbench_path: Optional[str],
) -> dict:
    import sys
    if vbench_path:
        sys.path.insert(0, vbench_path)
    from vbench import VBench

    my_VBench = VBench(device="cuda", video_path=video_path, output_path=output_dir)
    my_VBench.evaluate(
        videos_path=video_path,
        name="neuroadgen_eval",
        prompt_list=None,
        dimension_list=dims,
        local=False,
        read_frame=False,
    )

    results_file = Path(output_dir) / "neuroadgen_eval_eval_results.json"
    if results_file.exists():
        with open(results_file) as f:
            raw = json.load(f)
        return {k: v.get("video_results", 0.0) if isinstance(v, dict) else v
                for k, v in raw.items() if k in dims}
    return {}


def _vbench_cli(
    video_path: str,
    output_dir: str,
    dims: list[str],
    vbench_path: Optional[str],
) -> dict:
    cmd_prefix = [f"python {vbench_path}/evaluate.py"] if vbench_path else ["vbench"]
    cmd = [
        "python", "-m", "vbench.evaluate",
        "--videos_path", video_path,
        "--output_path", output_dir,
        "--dimension", *dims,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        logger.warning("VBench CLI failed: %s\n%s", exc.returncode, exc.stderr)
        return {d: 0.0 for d in dims}
    except FileNotFoundError:
        logger.warning("vbench not installed. Run: pip install vbench")
        return {d: 0.0 for d in dims}

    results_file = sorted(Path(output_dir).glob("*eval_results.json"))
    if not results_file:
        return {d: 0.0 for d in dims}
    with open(results_file[-1]) as f:
        raw = json.load(f)
    return {k: v.get("video_results", 0.0) if isinstance(v, dict) else v
            for k, v in raw.items() if k in dims}


def combined_score(
    brain_score: float,
    vbench_scores: dict,
    brain_weight: float = 0.6,
    quality_weight: float = 0.4,
) -> float:
    """
    Compute a combined NeuroAdGen score blending brain activation and video quality.

    The brain score captures neural engagement; VBench captures perceptual quality.
    A 60/40 split favours neuroscientific optimisation while penalising low-quality video.
    """
    quality_avg = float(sum(vbench_scores.values()) / max(len(vbench_scores), 1))
    return brain_weight * brain_score + quality_weight * quality_avg
