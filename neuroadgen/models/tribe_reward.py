"""
TribeV2 neural reward model wrapper for NeuroAdGen.

TribeV2 (facebook/tribev2) predicts cortical fMRI responses to video
across ~20k fsaverage5 mesh vertices. This module:

  1. Loads TribeV2 and runs predictions.
  2. Scores predicted activations within target brain ROIs.
  3. Supports three differentiability strategies:
       A. vjepa2_proxy  (default) — backprop through V-JEPA2 video features
          as a differentiable proxy for TribeV2's frozen encoder.
       B. reinforce      — REINFORCE / score-function estimator (black-box).
       C. surrogate_mlp  — lightweight MLP trained on (features→roi_scores).
  4. Accounts for the ~5-second hemodynamic lag in TribeV2 predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surrogate MLP (strategy C)
# ---------------------------------------------------------------------------

class SurrogateMLP(nn.Module):
    """
    Lightweight MLP that maps V-JEPA2 / mean-pooled video features
    to per-ROI scalar scores. Trained online from (features, TribeV2_scores)
    pairs so that gradients can flow back through it.
    """

    def __init__(self, in_dim: int, n_rois: int, hidden: list[int] | None = None) -> None:
        super().__init__()
        hidden = hidden or [512, 256, 128]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, n_rois))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main TribeReward class
# ---------------------------------------------------------------------------

class TribeReward(nn.Module):
    """
    Wraps TribeV2 + ROI scoring + differentiability strategies.

    Parameters
    ----------
    roi_config : dict
        Mapping of roi_name → {"vertices_mask_path": ..., "weight": ...}.
    differentiability_strategy : str
        One of "vjepa2_proxy", "reinforce", or "surrogate_mlp".
    cache_folder : str
        Local cache directory for model weights.
    hemodynamic_lag_sec : float
        Seconds of lag to skip at the start of TribeV2 predictions (default 5).
    prediction_end_sec : float
        End of usable prediction window in seconds (default 15).
    fps : int
        Frames-per-second of input video (used for window slicing).
    device : str | torch.device
        Compute device.
    """

    VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

    def __init__(
        self,
        roi_config: dict,
        differentiability_strategy: str = "vjepa2_proxy",
        cache_folder: str = "./cache",
        hemodynamic_lag_sec: float = 5.0,
        prediction_end_sec: float = 15.0,
        fps: int = 8,
        device: str | torch.device = "cuda",
        vjepa2_model_id: str | None = None,
        surrogate_mlp_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.roi_config = roi_config
        self.strategy = differentiability_strategy
        self.cache_folder = cache_folder
        self.lag_sec = hemodynamic_lag_sec
        self.end_sec = prediction_end_sec
        self.fps = fps
        self.device = torch.device(device)
        self.vjepa2_model_id = vjepa2_model_id or self.VJEPA2_MODEL_ID

        # ROI vertex masks (loaded lazily)
        self._roi_masks: dict[str, np.ndarray] = {}
        self._roi_weights: dict[str, float] = {}
        self._load_roi_masks()

        # TribeV2 (black-box, no gradient)
        self._tribe: Optional[object] = None

        # V-JEPA2 proxy encoder (for strategy A)
        self._vjepa2: Optional[nn.Module] = None
        self._vjepa2_proj: Optional[nn.Linear] = None  # feature dim → n_rois

        # Surrogate MLP (for strategy C)
        self._surrogate: Optional[SurrogateMLP] = None

        # Log active strategy
        logger.info("TribeReward initialised with strategy: %s", self.strategy)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_roi_masks(self) -> None:
        for name, cfg in self.roi_config.items():
            mask_path = cfg.get("vertices_mask_path", "")
            weight = float(cfg.get("weight", 1.0))
            self._roi_weights[name] = weight
            if mask_path and Path(mask_path).exists():
                self._roi_masks[name] = np.load(mask_path).astype(bool)
            else:
                # Placeholder — will use full vertex range; warn user
                logger.warning(
                    "ROI mask not found for '%s' at '%s'. "
                    "Run scripts/generate_roi_masks.py to create it. "
                    "Using all vertices as fallback.",
                    name,
                    mask_path,
                )
                self._roi_masks[name] = None

    def _ensure_tribe(self) -> None:
        if self._tribe is not None:
            return
        try:
            from tribev2 import TribeModel
            logger.info("Loading TribeV2 from cache: %s", self.cache_folder)
            self._tribe = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder=self.cache_folder,
            )
            logger.info("TribeV2 loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TribeV2: {exc}\n"
                "Ensure tribev2 is installed and numpy<2.1 is pinned."
            ) from exc

    def _ensure_vjepa2(self) -> None:
        if self._vjepa2 is not None:
            return
        try:
            from transformers import AutoVideoProcessor, AutoModel
            logger.info("Loading V-JEPA2 proxy encoder: %s", self.vjepa2_model_id)
            self._vjepa2_proc = AutoVideoProcessor.from_pretrained(
                self.vjepa2_model_id,
                cache_dir=self.cache_folder,
            )
            self._vjepa2 = AutoModel.from_pretrained(
                self.vjepa2_model_id,
                cache_dir=self.cache_folder,
            ).to(self.device).eval()
            feat_dim = self._vjepa2.config.hidden_size
            n_rois = len(self.roi_config)
            self._vjepa2_proj = nn.Linear(feat_dim, n_rois).to(self.device)
            logger.info("V-JEPA2 loaded (feature dim=%d, projecting to %d ROIs).", feat_dim, n_rois)
        except Exception as exc:
            logger.warning("V-JEPA2 load failed: %s — falling back to REINFORCE strategy.", exc)
            self.strategy = "reinforce"

    def _ensure_surrogate(self, in_dim: int) -> None:
        if self._surrogate is not None:
            return
        n_rois = len(self.roi_config)
        self._surrogate = SurrogateMLP(in_dim, n_rois).to(self.device)
        logger.info("Surrogate MLP initialised (%d→%d).", in_dim, n_rois)

    # ------------------------------------------------------------------
    # TribeV2 prediction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _tribe_predict(self, video_path: str) -> np.ndarray:
        """
        Run TribeV2 on a video file.

        Returns
        -------
        preds : np.ndarray  shape (n_timesteps, ~20480_vertices)
            Predicted cortical fMRI responses.
        """
        self._ensure_tribe()
        preds = self._tribe.predict(video_path)   # (T, V)
        logger.debug("TribeV2 raw prediction shape: %s", preds.shape)
        return preds

    def _slice_hrf_window(self, preds: np.ndarray, video_fps: int | None = None) -> np.ndarray:
        """
        Discard the first `lag_sec` TR window to account for hemodynamic lag.
        TribeV2 outputs one prediction per TR (assumed 1 s).
        For a 15 s video, usable window is TRs 5–15.
        """
        lag_tr = int(self.lag_sec)
        end_tr = int(self.end_sec)
        sliced = preds[lag_tr:end_tr]
        logger.debug("HRF-sliced predictions: TRs %d–%d → shape %s", lag_tr, end_tr, sliced.shape)
        return sliced

    def _compute_roi_scores(self, preds: np.ndarray) -> dict[str, float]:
        """Mean activation within each ROI vertex mask, averaged over TRs."""
        scores: dict[str, float] = {}
        for name, mask in self._roi_masks.items():
            if mask is not None and preds.shape[1] >= mask.shape[0]:
                roi_preds = preds[:, mask]
            else:
                roi_preds = preds  # fallback: all vertices
            scores[name] = float(roi_preds.mean())
        return scores

    # ------------------------------------------------------------------
    # Differentiable proxy (Strategy A: V-JEPA2)
    # ------------------------------------------------------------------

    def _vjepa2_roi_scores(self, video: Tensor) -> tuple[Tensor, dict[str, float]]:
        """
        Compute differentiable ROI scores via V-JEPA2 proxy.

        video : Tensor  (T, C, H, W)  in [0, 1]

        Returns
        -------
        diff_scores : Tensor  shape (n_rois,)  — gradients flow through this
        roi_dict    : dict[str, float]         — named scores for logging
        """
        self._ensure_vjepa2()

        # Resize frames to 224×224 expected by V-JEPA2
        import torch.nn.functional as F
        T, C, H, W = video.shape
        frames = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
        # (T, C, 224, 224) → (1, C, T, 224, 224)
        pixel_values = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type):
            features = self._vjepa2(pixel_values=pixel_values).last_hidden_state  # (1, L, D)
        pooled = features.mean(dim=1).squeeze(0)  # (D,)

        diff_scores = self._vjepa2_proj(pooled)  # (n_rois,)

        roi_names = list(self.roi_config.keys())
        roi_dict = {name: diff_scores[i].item() for i, name in enumerate(roi_names)}
        return diff_scores, roi_dict

    # ------------------------------------------------------------------
    # REINFORCE / score function estimator (Strategy B)
    # ------------------------------------------------------------------

    def _reinforce_roi_scores(
        self,
        video_path: str,
        log_prob: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Black-box reward with REINFORCE estimator.

        The gradient of E[R] w.r.t. θ is E[R · ∇_θ log p(x|θ)].
        Caller must supply `log_prob` = sum of log probs of the sampled latents.
        """
        preds = self._tribe_predict(video_path)
        preds = self._slice_hrf_window(preds)
        roi_dict = self._compute_roi_scores(preds)

        weights = torch.tensor(
            [self._roi_weights.get(n, 1.0) for n in roi_dict],
            device=self.device,
        )
        scores = torch.tensor(list(roi_dict.values()), device=self.device)
        R = (weights * scores).sum()

        # REINFORCE gradient surrogate: R * log_prob
        reinforce_loss = -R * log_prob
        return reinforce_loss, roi_dict

    # ------------------------------------------------------------------
    # Surrogate MLP (Strategy C)
    # ------------------------------------------------------------------

    def _surrogate_roi_scores(self, video: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Compute differentiable ROI scores via trained surrogate MLP."""
        import torch.nn.functional as F
        T, C, H, W = video.shape
        frames = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
        # Simple mean-pool over frames as feature
        feat = frames.mean(dim=0).reshape(-1)  # (C*H*W,) — or use V-JEPA2 if available

        self._ensure_surrogate(in_dim=feat.shape[0])
        diff_scores = self._surrogate(feat.to(self.device))  # (n_rois,)

        roi_names = list(self.roi_config.keys())
        roi_dict = {name: diff_scores[i].item() for i, name in enumerate(roi_names)}
        return diff_scores, roi_dict

    def update_surrogate(
        self,
        video: Tensor,
        tribe_scores: dict[str, float],
        lr: float = 1e-4,
    ) -> float:
        """Online update surrogate MLP on one (video, tribe_scores) pair."""
        import torch.nn.functional as F
        import torch.optim as optim

        T, C, H, W = video.shape
        frames = F.interpolate(video, size=(224, 224), mode="bilinear", align_corners=False)
        feat = frames.mean(dim=0).reshape(-1).detach()
        self._ensure_surrogate(in_dim=feat.shape[0])

        targets = torch.tensor(
            [tribe_scores.get(n, 0.0) for n in self.roi_config],
            device=self.device,
        )
        pred = self._surrogate(feat.to(self.device))
        loss = F.mse_loss(pred, targets)

        if not hasattr(self, "_surrogate_opt"):
            self._surrogate_opt = optim.Adam(self._surrogate.parameters(), lr=lr)
        self._surrogate_opt.zero_grad()
        loss.backward()
        self._surrogate_opt.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        video: Tensor,
        video_path: Optional[str] = None,
        log_prob: Optional[Tensor] = None,
        roi_weights: Optional[dict[str, float]] = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute composite brain reward for a video.

        Parameters
        ----------
        video      : Tensor (T, C, H, W) in [0, 1] — differentiable video tensor.
        video_path : str, optional — path to MP4 for black-box TribeV2 inference.
        log_prob   : Tensor, optional — required for REINFORCE strategy.
        roi_weights: dict, optional — override config weights.

        Returns
        -------
        reward     : Tensor (scalar) — differentiable composite reward.
        roi_scores : dict[str, float] — per-ROI scalar scores for logging.
        """
        weights = roi_weights or self._roi_weights

        if self.strategy == "vjepa2_proxy":
            diff_scores, roi_dict = self._vjepa2_roi_scores(video)
            w = torch.tensor(
                [weights.get(n, 1.0) for n in self.roi_config],
                device=self.device,
            )
            reward = (w * diff_scores).sum()

        elif self.strategy == "reinforce":
            if video_path is None or log_prob is None:
                raise ValueError("REINFORCE strategy requires video_path and log_prob.")
            reward, roi_dict = self._reinforce_roi_scores(video_path, log_prob)

        elif self.strategy == "surrogate_mlp":
            diff_scores, roi_dict = self._surrogate_roi_scores(video)
            w = torch.tensor(
                [weights.get(n, 1.0) for n in self.roi_config],
                device=self.device,
            )
            reward = (w * diff_scores).sum()

            # If ground-truth TribeV2 scores available, update surrogate
            if video_path is not None:
                try:
                    preds = self._tribe_predict(video_path)
                    preds = self._slice_hrf_window(preds)
                    tribe_scores = self._compute_roi_scores(preds)
                    loss = self.update_surrogate(video.detach(), tribe_scores)
                    logger.debug("Surrogate MLP update loss: %.4f", loss)
                except Exception as exc:
                    logger.warning("Surrogate update failed: %s", exc)

        else:
            raise ValueError(f"Unknown differentiability strategy: {self.strategy!r}")

        logger.debug("Composite reward: %.4f | ROI scores: %s", reward.item(), roi_dict)
        return reward, roi_dict

    @torch.no_grad()
    def score_video_file(self, video_path: str) -> dict[str, float]:
        """
        Score a video file via full TribeV2 (black-box, no gradient).
        Useful for final evaluation.
        """
        preds = self._tribe_predict(video_path)
        preds = self._slice_hrf_window(preds)
        return self._compute_roi_scores(preds)

    def get_full_vertex_predictions(self, video_path: str) -> np.ndarray:
        """Return raw per-vertex predictions for brain heatmap visualisation."""
        preds = self._tribe_predict(video_path)
        return self._slice_hrf_window(preds).mean(axis=0)  # (n_vertices,)
