# NeuroAdGen

**Brain-optimised ad video generation using TribeV2 neural reward alignment.**

NeuroAdGen takes an advertising brief and outputs a video advertisement iteratively optimised to maximise predicted cortical fMRI responses in target brain regions — visual cortex, emotional valence areas, memory circuits, and attention networks.

---

## How it works

```
Ad Brief
   │
   ▼
[LLM Prompt Expansion]
   │
   ▼
[Video Diffusion Model]  ←── LoRA fine-tuned on ad dataset (Stage 1)
   │  (Wan2.1-14B / HunyuanVideo / CogVideoX-5B)
   │
   ▼
[VADER-style Reward Loop]  ←── iterates N steps
   │
   ├── TribeV2 fMRI prediction  →  ROI activation scores
   │   (V-JEPA2 proxy for differentiability)
   │
   ├── Composite reward = Σ (weight × ROI_score) − KL_penalty
   │
   └── Backprop through last K=5 denoising steps → update LoRA weights
   │
   ▼
Optimised Ad Video  +  Brain Heatmap  +  ROI Scores JSON
```

---

## Quick start

### 1. Install

```bash
bash setup.sh
conda activate neuroadgen

# Generate ROI vertex masks (requires nilearn)
python neuroadgen/scripts/generate_roi_masks.py
```

### 2. Generate an ad

```bash
python neuroadgen/inference/generate.py \
  --brand "Nike" \
  --product "Air Max 2026" \
  --target_emotion "excitement and aspiration" \
  --scene "athlete running through neon-lit city at night" \
  --target_rois visual_engagement emotional_valence memory_encoding \
  --roi_weights 0.4 0.4 0.2 \
  --output_dir ./outputs/
```

### 3. Gradio web interface

```bash
python neuroadgen/app/gradio_app.py
# open http://localhost:7860
```

---

## Pipeline stages

### Stage 1 — LoRA fine-tuning (supervised)

Fine-tunes a video diffusion model on advertising video datasets to learn ad-specific visual styles and narrative structures.

```bash
# Single GPU
python neuroadgen/training/finetune_lora.py --config configs/default.yaml --data_dir data/ad_dataset

# Multi-GPU (DeepSpeed ZeRO-2)
accelerate launch --config_file configs/accelerate_zero2.yaml \
    neuroadgen/training/finetune_lora.py --config configs/default.yaml
```

### Stage 2 — Reward optimisation (VADER-style)

Iteratively adjusts LoRA weights using brain reward gradients from TribeV2 predictions.

```bash
python neuroadgen/training/reward_optimize.py \
  --config configs/default.yaml \
  --lora_ckpt checkpoints/lora_step_002000 \
  --prompts_file data/prompts/ad_prompts.txt
```

---

## VRAM requirements

| Model            | VRAM         | Notes                            |
|------------------|-------------|----------------------------------|
| Wan2.1-14B       | 40–80 GB    | Primary; multi-GPU recommended   |
| HunyuanVideo-13B | 40–60 GB    | Fallback                         |
| CogVideoX-5B     | 18–24 GB    | Single 4090 — use for dev        |
| + TribeV2        | +8 GB       | Neural reward model              |
| + V-JEPA2 proxy  | +4 GB       | Differentiable reward (Opt A)    |

For a single 24 GB GPU, set `video_model.name: "THUDM/CogVideoX-5b"` in `configs/default.yaml` and enable `hardware.cpu_offload: true`.

---

## Differentiability strategies

TribeV2 uses frozen encoders and cannot be directly backpropagated through. Three strategies are available:

| Strategy         | How                                                    | When to use                    |
|------------------|-------------------------------------------------------|-------------------------------|
| `vjepa2_proxy`   | Backprop through V-JEPA2 features as proxy            | Default — best gradient signal |
| `reinforce`      | REINFORCE score-function estimator (black-box reward) | If V-JEPA2 fails to load      |
| `surrogate_mlp`  | Lightweight MLP trained online on TribeV2 samples     | When proxy quality is poor    |

Set via `reward_opt.differentiability_strategy` in `configs/default.yaml`.

---

## Brain ROIs

TribeV2 predicts responses across ~20k vertices on the fsaverage5 cortical mesh. Five ROI targets are available:

| ROI                     | Brain area                    | Ad effect                        |
|-------------------------|------------------------------|----------------------------------|
| `visual_engagement`     | V1, V2, MT+                  | Vivid dynamic imagery            |
| `emotional_valence`     | vmPFC, temporal pole         | Emotional resonance              |
| `attention_capture`     | TPJ, IPS                     | Attentional salience             |
| `narrative_comprehension` | Broca, Wernicke, DMN       | Story absorption                 |
| `memory_encoding`       | Parahippocampal cortex       | Ad memorability                  |

ROI vertex masks are generated from the Destrieux atlas projected to fsaverage5. Run `python neuroadgen/scripts/generate_roi_masks.py` to build them.

---

## Output structure

```
outputs/
├── Nike_Air_Max_2026_<timestamp>_initial.mp4      # pre-optimisation video
├── Nike_Air_Max_2026_<timestamp>_optimized.mp4    # brain-optimised video
├── Nike_Air_Max_2026_<timestamp>_brain_heatmap.png
├── Nike_Air_Max_2026_<timestamp>_brain_interactive.html
└── Nike_Air_Max_2026_<timestamp>_results.json
```

`results.json`:
```json
{
  "video_path": "outputs/..._optimized.mp4",
  "brain_heatmap_path": "outputs/..._brain_heatmap.png",
  "roi_scores": {
    "visual_engagement": 0.72,
    "emotional_valence": 0.68,
    "memory_encoding": 0.51
  },
  "composite_score": 0.66,
  "optimization_trajectory": [...]
}
```

---

## Seedance 2.0

`SeedanceAPIGenerator` wraps the fal.ai Seedance 2.0 API for high-quality initial draft generation. Set `FAL_KEY` in your environment.

```python
from neuroadgen.models.video_gen import SeedanceAPIGenerator
gen = SeedanceAPIGenerator()
video_path = gen.generate("athlete running in neon city")
```

**Note:** Seedance weights are proprietary (ByteDance). Reward gradient optimisation cannot flow through the API — switch to `VideoGenerator` for Stage 2.

---

## Key references

- [TribeV2](https://github.com/facebookresearch/tribev2) — neural encoding model for video
- [VADER](https://github.com/mihirp1998/VADER) — reward gradient alignment for video diffusion
- [Wan2.1](https://github.com/Wan-AI/Wan2.1) — primary video generation backbone
- [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) — fallback backbone
- [CogVideoX](https://github.com/zai-org/CogVideo) — single-GPU fallback
- [finetrainers](https://github.com/huggingface/finetrainers) — HuggingFace fine-tuning toolkit
- [Seedance 2.0 API](https://github.com/Anil-matcha/Seedance-2.0-API) — draft generation

---

## ⚠️ Ethics

**NeuroAdGen uses a computational model (TribeV2) trained on consented fMRI data from human participants. It does not access real-time brain signals, implants, or any live neural data.**

This system is intended for:
- Research into neural correlates of advertising effectiveness.
- Academic study of neuroaesthetics and media cognition.
- Transparent, disclosed commercial applications where participants are informed.

**This system must NOT be used for:**
- Subliminal advertising or covert psychological manipulation.
- Targeting vulnerable populations without safeguards.
- Any application where the neural optimisation is concealed from viewers.

Deployment in consumer-facing advertising contexts requires compliance with applicable regulations (FTC guidelines, EU AI Act, etc.) and full disclosure to audiences.

---

## Running tests

```bash
# Unit tests (no GPU required)
pytest neuroadgen/tests/test_pipeline.py -v

# Integration tests (requires GPU + model downloads)
pytest neuroadgen/tests/test_pipeline.py -v -m integration
```
