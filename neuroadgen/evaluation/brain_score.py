"""
Brain score evaluation utilities.

Provides functions to:
  - Score a video file against TribeV2 predictions.
  - Compare pre- vs post-optimisation brain activations.
  - Compute per-ROI delta scores and statistical summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def score_video(
    video_path: str,
    roi_config: dict,
    cfg: dict,
    device: str = "cuda",
) -> dict:
    """
    Score a single video file using TribeV2.

    Returns
    -------
    dict with keys:
        roi_scores: dict[str, float]
        composite_score: float
        vertex_predictions: np.ndarray (n_vertices,)
    """
    from neuroadgen.models.tribe_reward import TribeReward

    reward_model = TribeReward(
        roi_config=roi_config,
        differentiability_strategy="reinforce",  # black-box scoring only
        cache_folder=cfg["tribe"]["cache_folder"],
        hemodynamic_lag_sec=cfg["tribe"]["hemodynamic_lag_sec"],
        prediction_end_sec=cfg["tribe"]["prediction_end_sec"],
        fps=cfg["video_model"]["fps"],
        device=device,
    )

    roi_scores = reward_model.score_video_file(video_path)
    vertex_preds = reward_model.get_full_vertex_predictions(video_path)

    weights = np.array([roi_config[k].get("weight", 1.0) for k in roi_scores])
    values = np.array(list(roi_scores.values()))
    composite = float((weights * values).sum() / weights.sum())

    return {
        "roi_scores": roi_scores,
        "composite_score": composite,
        "vertex_predictions": vertex_preds,
    }


def compare_videos(
    video_before: str,
    video_after: str,
    roi_config: dict,
    cfg: dict,
    device: str = "cuda",
) -> dict:
    """
    Compare brain activation scores before and after optimisation.

    Returns
    -------
    dict with before/after scores and per-ROI delta.
    """
    before = score_video(video_before, roi_config, cfg, device)
    after = score_video(video_after, roi_config, cfg, device)

    delta_roi = {
        k: after["roi_scores"].get(k, 0) - before["roi_scores"].get(k, 0)
        for k in roi_config
    }
    delta_composite = after["composite_score"] - before["composite_score"]
    pct_improvement = 100.0 * delta_composite / (abs(before["composite_score"]) + 1e-8)

    return {
        "before": before,
        "after": after,
        "delta_roi": delta_roi,
        "delta_composite": delta_composite,
        "pct_improvement": pct_improvement,
    }


def save_evaluation_report(results: dict, output_path: str) -> None:
    """Save evaluation results to a JSON report file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(results)
    with open(out, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Evaluation report saved: %s", out)
