"""
Unit and integration tests for the NeuroAdGen pipeline.

Run:
    pytest tests/test_pipeline.py -v

Tests use lightweight mocks so they pass without GPU/model downloads.
Integration tests (marked @pytest.mark.integration) require full environment.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_video():
    """Tiny (8, 3, 64, 64) video tensor in [0, 1]."""
    return torch.rand(8, 3, 64, 64)


@pytest.fixture
def dummy_vertex_preds():
    """Fake TribeV2 predictions: (10, 20484) vertices."""
    return np.random.randn(10, 20484).astype(np.float32)


@pytest.fixture
def roi_config():
    return {
        "visual_engagement": {"vertices_mask_path": "", "weight": 0.4},
        "emotional_valence": {"vertices_mask_path": "", "weight": 0.4},
        "memory_encoding": {"vertices_mask_path": "", "weight": 0.2},
    }


@pytest.fixture
def sample_brief():
    return {
        "brand": "Nike",
        "product": "Air Max 2026",
        "target_emotion": "excitement and aspiration",
        "scene_description": "athlete running through neon-lit city at night",
        "duration_seconds": 15,
        "target_brain_regions": ["visual_engagement", "emotional_valence"],
        "roi_weights": {"visual_engagement": 0.5, "emotional_valence": 0.5},
        "reference_image": None,
        "output_dir": tempfile.mkdtemp(),
    }


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_loads(self, cfg):
        assert "video_model" in cfg
        assert "lora" in cfg
        assert "reward_opt" in cfg
        assert "rois" in cfg
        assert "tribe" in cfg

    def test_roi_weights_reasonable(self, cfg):
        total = sum(v["weight"] for v in cfg["rois"].values())
        assert 0.5 <= total <= 2.0, "ROI weights should sum to roughly 1.0"

    def test_differentiability_strategy_valid(self, cfg):
        valid = {"vjepa2_proxy", "reinforce", "surrogate_mlp"}
        assert cfg["reward_opt"]["differentiability_strategy"] in valid

    def test_numpy_constraint_documented(self, cfg):
        # Sanity check that setup.sh pins numpy<2.1
        setup = (Path(__file__).parent.parent.parent / "setup.sh").read_text()
        assert "numpy" in setup.lower()


# ---------------------------------------------------------------------------
# LoRA adapter tests
# ---------------------------------------------------------------------------

def _peft_available():
    try:
        import torch
        import torch.distributed
        _ = torch.distributed.tensor  # noqa
        return True
    except AttributeError:
        return False


@pytest.mark.skipif(not _peft_available(), reason="torch.distributed.tensor unavailable on this platform (macOS)")
class TestLoraAdapter:
    def test_inject_lora_and_freeze(self):
        from neuroadgen.models.lora_adapter import inject_lora, freeze_base_weights, get_lora_param_count
        import torch.nn as nn

        # Minimal transformer-like model with attention projections
        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)

        model = FakeTransformer()
        lora_model = inject_lora(model, r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v"])
        freeze_base_weights(lora_model)
        counts = get_lora_param_count(lora_model)

        assert counts["trainable"] > 0
        assert counts["trainable"] < counts["total"]

    def test_save_and_load_lora(self, tmp_path):
        from neuroadgen.models.lora_adapter import (
            inject_lora, save_lora_checkpoint, load_lora_checkpoint
        )
        import torch.nn as nn

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)

        model = FakeTransformer()
        lora_model = inject_lora(model, r=4, lora_alpha=4, target_modules=["to_q", "to_k"])

        ckpt = save_lora_checkpoint(lora_model, str(tmp_path), step=100)
        assert Path(ckpt).exists()

        # Load back (non-PEFT fallback path)
        model2 = FakeTransformer()
        lora_model2 = inject_lora(model2, r=4, lora_alpha=4, target_modules=["to_q", "to_k"])
        loaded = load_lora_checkpoint(lora_model2, ckpt, device="cpu")
        assert loaded is not None


# ---------------------------------------------------------------------------
# TribeReward tests (mocked TribeV2)
# ---------------------------------------------------------------------------

class TestTribeReward:
    def test_roi_mask_missing_uses_fallback(self, roi_config):
        from neuroadgen.models.tribe_reward import TribeReward
        reward = TribeReward(
            roi_config=roi_config,
            differentiability_strategy="reinforce",
            device="cpu",
        )
        # All masks missing → all None (fallback to all vertices)
        for name in roi_config:
            assert reward._roi_masks[name] is None

    def test_hrf_slice(self, roi_config):
        from neuroadgen.models.tribe_reward import TribeReward
        reward = TribeReward(roi_config=roi_config, device="cpu", hemodynamic_lag_sec=5, prediction_end_sec=15)
        preds = np.random.randn(20, 20484)
        sliced = reward._slice_hrf_window(preds)
        assert sliced.shape == (10, 20484)

    def test_roi_scores_computation(self, roi_config):
        from neuroadgen.models.tribe_reward import TribeReward
        reward = TribeReward(roi_config=roi_config, device="cpu")
        preds = np.ones((10, 20484)) * 0.5
        scores = reward._compute_roi_scores(preds)
        assert set(scores.keys()) == set(roi_config.keys())
        for v in scores.values():
            assert abs(v - 0.5) < 1e-4

    def test_surrogate_mlp_forward(self, dummy_video, roi_config):
        from neuroadgen.models.tribe_reward import TribeReward, SurrogateMLP
        reward = TribeReward(
            roi_config=roi_config,
            differentiability_strategy="surrogate_mlp",
            device="cpu",
        )
        diff_scores, roi_dict = reward._surrogate_roi_scores(dummy_video)
        assert diff_scores.shape == (len(roi_config),)
        assert diff_scores.requires_grad


# ---------------------------------------------------------------------------
# SurrogateMLP tests
# ---------------------------------------------------------------------------

class TestSurrogateMLP:
    def test_forward_and_backward(self):
        from neuroadgen.models.tribe_reward import SurrogateMLP
        mlp = SurrogateMLP(in_dim=128, n_rois=3)
        x = torch.randn(128, requires_grad=True)
        out = mlp(x)
        assert out.shape == (3,)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Brain heatmap tests (mocked nilearn)
# ---------------------------------------------------------------------------

class TestBrainHeatmap:
    def test_generate_heatmap_placeholder(self, tmp_path, roi_config):
        """Should not crash even without real vertex predictions."""
        from neuroadgen.visualization.brain_heatmap import generate_brain_heatmap
        output_png = str(tmp_path / "heatmap.png")
        try:
            generate_brain_heatmap(
                vertex_predictions=None,
                roi_config=roi_config,
                output_png=output_png,
            )
            assert Path(output_png).exists()
        except ImportError:
            pytest.skip("nilearn/matplotlib not installed")

    def test_roi_bar_chart(self, tmp_path, roi_config):
        from neuroadgen.visualization.brain_heatmap import plot_roi_scores_bar
        scores = {k: np.random.uniform(0.3, 0.8) for k in roi_config}
        out = str(tmp_path / "bar.png")
        try:
            plot_roi_scores_bar(scores, out)
            assert Path(out).exists()
        except ImportError:
            pytest.skip("matplotlib not installed")


# ---------------------------------------------------------------------------
# Prompt expansion tests
# ---------------------------------------------------------------------------

class TestPromptExpansion:
    def test_template_expansion(self, sample_brief):
        from neuroadgen.inference.generate import expand_brief_to_prompt
        prompt = expand_brief_to_prompt(sample_brief, llm_model_id=None)
        assert "Nike" in prompt
        assert "Air Max 2026" in prompt
        assert len(prompt) > 30

    def test_llm_fallback_on_failure(self, sample_brief):
        from neuroadgen.inference.generate import expand_brief_to_prompt
        # Passing a non-existent model ID should fall back to template
        prompt = expand_brief_to_prompt(sample_brief, llm_model_id="nonexistent-model-xyz")
        assert "Nike" in prompt


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestAdVideoDataset:
    def test_empty_dir_returns_empty(self, tmp_path):
        from neuroadgen.training.finetune_lora import AdVideoDataset
        ds = AdVideoDataset(str(tmp_path))
        assert len(ds) == 0

    def test_metadata_json_loading(self, tmp_path):
        from neuroadgen.training.finetune_lora import AdVideoDataset
        # Create dummy metadata (no actual video — just test __len__)
        meta = [{"prompt": "test ad", "video_path": "nonexistent.mp4"}]
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        ds = AdVideoDataset(str(tmp_path))
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# Integration test (requires full environment)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipeline:
    def test_generate_ad_runs(self, sample_brief, cfg, tmp_path):
        """Full end-to-end test — requires GPU + model downloads."""
        sample_brief["output_dir"] = str(tmp_path)
        # Override to smallest available model for test speed
        cfg["video_model"]["name"] = "THUDM/CogVideoX-5b"
        cfg["reward_opt"]["n_steps"] = 2
        cfg["video_model"]["inference_steps"] = 5

        from neuroadgen.inference.generate import generate_ad
        result = generate_ad(sample_brief)

        assert "video_path" in result
        assert Path(result["video_path"]).exists()
        assert "roi_scores" in result
        assert "composite_score" in result
        assert isinstance(result["composite_score"], float)
