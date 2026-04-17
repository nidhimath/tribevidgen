"""
LoRA adapter injection and checkpoint management for NeuroAdGen.

Injects LoRA adapters into the cross-attention layers of the video DiT
backbone using HuggingFace PEFT, and provides save/load helpers for
LoRA-only checkpoint management (much smaller than full model saves).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 64,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "lora_dropout": 0.1,
    "bias": "none",
}


def inject_lora(
    model: nn.Module,
    r: int = 64,
    lora_alpha: int = 64,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    Inject LoRA adapters into a video DiT model using HuggingFace PEFT.

    Parameters
    ----------
    model          : The DiT transformer module to adapt.
    r              : LoRA rank.
    lora_alpha     : LoRA alpha scaling factor.
    target_modules : List of submodule name patterns to target.
    lora_dropout   : Dropout applied inside LoRA branch.

    Returns
    -------
    The same model with LoRA adapters injected (PEFT LoraModel).
    """
    from peft import LoraConfig, get_peft_model

    modules = target_modules or DEFAULT_LORA_CONFIG["target_modules"]
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        # task_type not set because this is a custom diffusion module
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model


def freeze_base_weights(model: nn.Module) -> None:
    """Freeze all parameters except LoRA layers."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)


def save_lora_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    step: int,
    metadata: dict | None = None,
) -> str:
    """
    Save LoRA adapter weights + metadata to a checkpoint directory.

    Saves only the LoRA delta weights (not the frozen base model),
    keeping checkpoint files small (~100–500 MB vs 28+ GB for full model).

    Returns
    -------
    str : Path to the saved checkpoint directory.
    """
    from peft import PeftModel

    ckpt_path = Path(checkpoint_dir) / f"lora_step_{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        model.save_pretrained(str(ckpt_path))
        logger.info("LoRA checkpoint saved (PEFT format): %s", ckpt_path)
    else:
        # Fallback: save only lora_ parameters as a state_dict
        lora_state = {
            k: v for k, v in model.state_dict().items() if "lora_" in k
        }
        torch.save(lora_state, ckpt_path / "lora_weights.pt")
        logger.info("LoRA weights saved (raw state_dict): %s", ckpt_path)

    # Save metadata
    meta = {"step": step, **(metadata or {})}
    with open(ckpt_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(ckpt_path)


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """
    Load LoRA adapter weights from a checkpoint directory into `model`.

    Supports both PEFT-format saves and raw state_dict saves.

    Returns
    -------
    The model with LoRA weights loaded (in-place modification + return).
    """
    from peft import PeftModel

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    peft_config_file = ckpt_path / "adapter_config.json"
    if peft_config_file.exists():
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.to(device)
        logger.info("LoRA checkpoint loaded (PEFT format) from: %s", ckpt_path)
    else:
        lora_weights_file = ckpt_path / "lora_weights.pt"
        if not lora_weights_file.exists():
            raise FileNotFoundError(f"No LoRA weights found in: {ckpt_path}")
        lora_state = torch.load(lora_weights_file, map_location=device)
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        if unexpected:
            logger.warning("Unexpected keys when loading LoRA: %s", unexpected[:5])
        logger.info("LoRA weights loaded (raw state_dict) from: %s", ckpt_path)

    return model


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """
    Merge LoRA delta weights into the base model weights in-place.
    Use before export/serving to avoid PEFT overhead at inference time.
    """
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        logger.warning("merge_lora_into_base called on non-PeftModel — no-op.")
        return model
    merged = model.merge_and_unload()
    logger.info("LoRA weights merged into base model.")
    return merged


def get_lora_param_count(model: nn.Module) -> dict[str, int]:
    """Return trainable vs total param counts for logging."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}
