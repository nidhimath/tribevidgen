"""
Brain activation heatmap visualisation using nilearn + TribeV2 predictions.

Generates:
  1. Static PNG with left/right hemisphere views coloured by ROI activation.
  2. Interactive HTML brain viewer via nilearn.plotting.view_surf.

TribeV2 predictions are on the fsaverage5 mesh (~20k vertices split
~10k per hemisphere). This module projects per-vertex scalar activations
onto the cortical surface and renders them with annotated ROI boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# fsaverage5 has 10242 vertices per hemisphere (left + right = 20484 total)
N_VERTICES_PER_HEMI = 10242


def generate_brain_heatmap(
    vertex_predictions: Optional[np.ndarray],
    roi_config: dict,
    output_png: str,
    output_html: Optional[str] = None,
    title: str = "NeuroAdGen Brain Activation Map",
    colormap: str = "hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Project vertex-level TribeV2 predictions onto the fsaverage5 surface
    and save a static PNG + optional interactive HTML.

    Parameters
    ----------
    vertex_predictions : np.ndarray, shape (20484,) or None
        Mean-over-time predicted cortical activations from TribeV2.
        If None, generates a placeholder figure.
    roi_config : dict
        ROI name → {"vertices_mask_path": ..., "weight": ...}.
    output_png : str
        Output path for the static PNG.
    output_html : str, optional
        Output path for the interactive HTML viewer.
    title : str
        Figure title.
    colormap : str
        Matplotlib colormap for activation values.
    vmin, vmax : float, optional
        Colorbar range; auto-computed if None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from nilearn import plotting, datasets, surface
    except ImportError as exc:
        raise ImportError(
            "nilearn and matplotlib are required for brain heatmaps. "
            "Run: pip install nilearn matplotlib"
        ) from exc

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load fsaverage5 surface mesh
    # ------------------------------------------------------------------
    logger.info("Loading fsaverage5 surface mesh...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    # ------------------------------------------------------------------
    # Prepare vertex data
    # ------------------------------------------------------------------
    if vertex_predictions is None:
        logger.warning("No vertex predictions provided — generating placeholder heatmap.")
        vertex_predictions = np.zeros(N_VERTICES_PER_HEMI * 2)

    # Pad/trim to expected size
    expected_n = N_VERTICES_PER_HEMI * 2
    if vertex_predictions.shape[0] < expected_n:
        padded = np.zeros(expected_n)
        padded[:vertex_predictions.shape[0]] = vertex_predictions
        vertex_predictions = padded
    else:
        vertex_predictions = vertex_predictions[:expected_n]

    lh_data = vertex_predictions[:N_VERTICES_PER_HEMI]
    rh_data = vertex_predictions[N_VERTICES_PER_HEMI:]

    # Auto colorbar range
    if vmin is None:
        vmin = float(np.percentile(vertex_predictions, 5))
    if vmax is None:
        vmax = float(np.percentile(vertex_predictions, 95))

    # ------------------------------------------------------------------
    # Overlay ROI highlights
    # ------------------------------------------------------------------
    lh_roi_overlay = np.zeros(N_VERTICES_PER_HEMI)
    rh_roi_overlay = np.zeros(N_VERTICES_PER_HEMI)
    roi_labels: list[str] = []
    roi_colors = _get_roi_colors(len(roi_config))

    for i, (roi_name, roi_cfg) in enumerate(roi_config.items()):
        mask_path = roi_cfg.get("vertices_mask_path", "")
        if mask_path and Path(mask_path).exists():
            mask = np.load(mask_path).astype(bool)
            if mask.shape[0] == N_VERTICES_PER_HEMI * 2:
                lh_mask = mask[:N_VERTICES_PER_HEMI]
                rh_mask = mask[N_VERTICES_PER_HEMI:]
            elif mask.shape[0] == N_VERTICES_PER_HEMI:
                lh_mask = mask
                rh_mask = mask
            else:
                logger.warning("Unexpected mask shape %s for ROI '%s'", mask.shape, roi_name)
                continue
            # Mark ROI vertices with a distinct value (above vmax for visibility)
            lh_roi_overlay[lh_mask] = vmax * 1.5 * (i + 1) / len(roi_config)
            rh_roi_overlay[rh_mask] = vmax * 1.5 * (i + 1) / len(roi_config)
            roi_labels.append(roi_name)

    # ------------------------------------------------------------------
    # Static PNG: 4-panel figure (LH lateral, LH medial, RH lateral, RH medial)
    # ------------------------------------------------------------------
    logger.info("Generating static brain heatmap PNG...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={"projection": "3d"})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    views = [
        ("left",  "lateral", fsaverage["pial_left"],  lh_data, axes[0, 0]),
        ("left",  "medial",  fsaverage["pial_left"],  lh_data, axes[0, 1]),
        ("right", "lateral", fsaverage["pial_right"], rh_data, axes[1, 0]),
        ("right", "medial",  fsaverage["pial_right"], rh_data, axes[1, 1]),
    ]

    for hemi, view, mesh, data, ax in views:
        plotting.plot_surf_stat_map(
            mesh,
            stat_map=data,
            hemi=hemi,
            view=view,
            colorbar=False,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi.upper()} — {view}", fontsize=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label("Predicted fMRI Response (a.u.)", fontsize=10)

    # ROI legend
    if roi_labels:
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=roi_colors[i], label=roi_labels[i])
            for i in range(len(roi_labels))
        ]
        fig.legend(handles=legend_patches, loc="lower right", title="Target ROIs",
                   fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Brain heatmap PNG saved: %s", output_png)

    # ------------------------------------------------------------------
    # Interactive HTML viewer
    # ------------------------------------------------------------------
    if output_html:
        _generate_interactive_viewer(
            fsaverage=fsaverage,
            lh_data=lh_data,
            rh_data=rh_data,
            output_html=output_html,
            title=title,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
        )


def _generate_interactive_viewer(
    fsaverage,
    lh_data: np.ndarray,
    rh_data: np.ndarray,
    output_html: str,
    title: str,
    colormap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Generate an interactive HTML brain viewer using nilearn.plotting.view_surf."""
    try:
        from nilearn import plotting
    except ImportError:
        logger.warning("nilearn not available for interactive viewer.")
        return

    try:
        lh_view = plotting.view_surf(
            surf_mesh=fsaverage["pial_left"],
            surf_map=lh_data,
            bg_map=fsaverage["sulc_left"],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            title=f"{title} — Left Hemisphere",
        )

        rh_view = plotting.view_surf(
            surf_mesh=fsaverage["pial_right"],
            surf_map=rh_data,
            bg_map=fsaverage["sulc_right"],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            title=f"{title} — Right Hemisphere",
        )

        # Combine both hemisphere views into one HTML file
        lh_html = lh_view.get_iframe(width="48%", height="500px")
        rh_html = rh_view.get_iframe(width="48%", height="500px")

        combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ text-align: center; color: #e94560; }}
        .container {{ display: flex; gap: 2%; justify-content: center; }}
        .hemisphere {{ text-align: center; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="container">
        <div class="hemisphere">
            <h3>Left Hemisphere</h3>
            {lh_html}
        </div>
        <div class="hemisphere">
            <h3>Right Hemisphere</h3>
            {rh_html}
        </div>
    </div>
</body>
</html>"""

        Path(output_html).write_text(combined_html)
        logger.info("Interactive brain viewer saved: %s", output_html)

    except Exception as exc:
        logger.warning("Interactive viewer generation failed: %s", exc)


def _get_roi_colors(n: int) -> list:
    """Return a list of n distinct colors for ROI annotation."""
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap("tab10").resampled(n)
    return [cmap(i) for i in range(n)]


def plot_roi_scores_bar(
    roi_scores: dict[str, float],
    output_path: str,
    title: str = "Brain ROI Activation Scores",
) -> None:
    """Generate a horizontal bar chart of ROI scores for the Gradio interface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(roi_scores.keys())
    values = [roi_scores[n] for n in names]
    colors = _get_roi_colors(len(names))

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.7)))
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("Predicted fMRI Activation (normalised)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2 if values else 1.0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROI scores bar chart saved: %s", output_path)
