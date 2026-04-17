"""
Generate fsaverage5 ROI vertex masks from the HCP MMP1.0 parcellation.

These masks are required by TribeReward to compute per-ROI activation scores
from TribeV2's ~20k-vertex fsaverage5 predictions.

Approach:
  1. Fetch HCP MMP1.0 atlas labels projected onto fsaverage5 via nilearn.
  2. For each ROI in tribe_rois.yaml, collect vertex indices matching
     the named parcels and save as a boolean .npy mask.

Run:
    python scripts/generate_roi_masks.py
    python scripts/generate_roi_masks.py --atlas destrieux  # fallback atlas

Output masks are saved to configs/masks/ and referenced by default.yaml.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

MASKS_DIR = Path(__file__).parent.parent / "configs" / "masks"
CONFIG_DIR = Path(__file__).parent.parent / "configs"
N_VERTICES_PER_HEMI = 10242  # fsaverage5


def fetch_fsaverage5_labels(atlas: str = "destrieux") -> tuple[np.ndarray, list[str]]:
    """
    Fetch surface parcellation labels for fsaverage5.

    Returns
    -------
    labels : np.ndarray, shape (20484,)
        Integer label index per vertex (left hemi 0:10242, right hemi 10242:20484).
    label_names : list[str]
        Human-readable names indexed by integer label.
    """
    from nilearn import datasets, surface

    if atlas == "destrieux":
        data = datasets.fetch_atlas_surf_destrieux()
        lh_labels = np.array(data["map_left"]).astype(int)
        rh_labels = np.array(data["map_right"]).astype(int)
        label_names = [s.decode("utf-8") if isinstance(s, bytes) else s
                       for s in data["labels"]]
    elif atlas == "hcp_mmp1":
        # HCP MMP1.0 requires manual download — use Destrieux as approximation
        logger.warning("HCP MMP1.0 auto-download not available via nilearn. Using Destrieux as proxy.")
        return fetch_fsaverage5_labels("destrieux")
    else:
        raise ValueError(f"Unknown atlas: {atlas!r}. Use 'destrieux' or 'hcp_mmp1'.")

    if lh_labels.shape[0] != N_VERTICES_PER_HEMI:
        logger.warning("Label array size mismatch (%d vs %d expected)", lh_labels.shape[0], N_VERTICES_PER_HEMI)

    labels = np.concatenate([lh_labels, rh_labels])
    return labels, label_names


def build_roi_mask(
    labels: np.ndarray,
    label_names: list[str],
    target_region_patterns: list[str],
) -> np.ndarray:
    """
    Build a boolean vertex mask for a set of region name patterns.

    Parameters
    ----------
    labels         : Integer label array, shape (20484,).
    label_names    : Name string for each label index.
    target_region_patterns : List of regex/substring patterns to match label names.

    Returns
    -------
    mask : np.ndarray, bool, shape (20484,).
    """
    matched_indices = set()
    for pattern in target_region_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for idx, name in enumerate(label_names):
            if regex.search(name):
                matched_indices.add(idx)

    if not matched_indices:
        logger.warning("No label names matched patterns: %s", target_region_patterns)
        logger.info("Available labels: %s", label_names[:20])

    mask = np.isin(labels, list(matched_indices))
    logger.info("Mask covers %d / %d vertices", mask.sum(), len(mask))
    return mask


# ROI → Destrieux label patterns mapping
# These patterns match substrings/regex in Destrieux atlas region names.
ROI_PATTERNS = {
    "visual_engagement": [
        r"G_and_S_occipital",       # V1/V2 proxy
        r"Pole_occipital",
        r"G_occipital",
        r"S_occipital",
        r"G_cuneus",                # V1
        r"S_calcarine",             # V1 (calcarine sulcus)
        r"G_and_S_cuneus",
        r"Lat_Fis.*Post",           # lateral occipital / MT proxy
    ],
    "emotional_valence": [
        r"G_orbital",               # OFC / vmPFC
        r"G_rectus",                # vmPFC
        r"G_and_S_cingul.*Ant",     # anterior cingulate / vmPFC
        r"Pole_temporal",           # temporal pole / amygdala-adjacent
        r"G_temporal_inf",          # inferior temporal (emotion)
    ],
    "attention_capture": [
        r"G_parietal_sup",          # superior parietal / IPS
        r"S_intrapariet",           # intraparietal sulcus
        r"G_and_S_subcentral",      # TPJ proxy
        r"G_temp_sup.*Banks",       # TPJ / superior temporal junction
        r"S_temporal_sup",
    ],
    "narrative_comprehension": [
        r"G_front_inf.*Triangul",   # Broca (BA45)
        r"G_front_inf.*Opercular",  # Broca (BA44)
        r"G_temporal_sup",          # Wernicke (superior temporal)
        r"S_temporal_sup",
        r"G_cingul.*Post",          # PCC / DMN
        r"G_precuneus",             # precuneus / DMN
        r"S_cingul",
    ],
    "memory_encoding": [
        r"G_oc.*temp.*med.*Lingual",# lingual gyrus / PHC
        r"G_and_S_parahippocampal", # parahippocampal
        r"G_oc.*temp.*lat",         # fusiform / parahippocampal
        r"S_oc.*temp.*med",
    ],
}


def generate_all_masks(atlas: str = "destrieux") -> None:
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching fsaverage5 %s labels...", atlas)
    labels, label_names = fetch_fsaverage5_labels(atlas)

    roi_yaml_path = CONFIG_DIR / "tribe_rois.yaml"
    with open(roi_yaml_path) as f:
        tribe_cfg = yaml.safe_load(f)

    for roi_name, patterns in ROI_PATTERNS.items():
        logger.info("Building mask for ROI: %s", roi_name)
        mask = build_roi_mask(labels, label_names, patterns)

        out_path = MASKS_DIR / f"{roi_name.replace('_', '')}.npy"
        # Use canonical filenames matching default.yaml
        name_map = {
            "visual_engagement": "v1_v2_mt.npy",
            "emotional_valence": "vmpfc_amygdala.npy",
            "attention_capture": "tpj_ips.npy",
            "narrative_comprehension": "broca_wernicke_dmn.npy",
            "memory_encoding": "hippocampal.npy",
        }
        out_path = MASKS_DIR / name_map.get(roi_name, f"{roi_name}.npy")
        np.save(out_path, mask)
        logger.info("  Saved %s  (vertices=%d)", out_path.name, mask.sum())

    logger.info("All ROI masks generated in %s", MASKS_DIR)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="destrieux", choices=["destrieux", "hcp_mmp1"])
    args = p.parse_args()
    generate_all_masks(atlas=args.atlas)
