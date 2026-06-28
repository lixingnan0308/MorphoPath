"""Per-slide concept-overlay interpretability visualisation for MorphoPath (UNI-512).

Renders, for one whole-slide image, how each diagnostic concept is distributed
across patches — the model's interpretable evidence for its 1p/19q prediction.
Use ``--label TP`` (true-positive / codeleted) to render the oligo concepts, or
``--label TN`` (true-negative) to render the astrocytoma concept.

OVERLAP MODE:
  - Per-concept maps drop the argmax (`category == k`) restriction. Every
    diagnostic patch is coloured by `concept_share[i, k]` for THIS concept,
    where concept_share = attn[i, :] / sum_concepts(attn[i, :]) — a LINEAR
    per-patch concept proportion. Softmax was tried first but saturates to
    uniform 1/K because attn values are ~1e-3; linear normalisation
    preserves the multiplicative ratios between concepts.
  - Top-N per concept ranks ALL diagnostic patches by concept_share[:, k].
    Patches with uniform attn across concepts get share ~0.25 each and never
    dominate any concept's top-N; concept-specific patches dominate their
    preferred concept; genuine multi-concept patches appear in multiple top-Ns.
  - Colorbar label: "Concept share" (in [0, 1], rows sum to 1).
  - coords.csv adds columns:
      concept_share    (float) — softmax(attn, axis=1)[i, k]
      is_argmax_winner (bool)  — True if this patch's argmax winner is this concept
      argmax_concept   (int)   — the actual argmax concept index (0-5)
  - concept_label_heatmap.pdf (3-class partition figure) STILL uses argmax —
    that figure's semantic IS "where does this patch finally belong".
  - concept_count_bar / contribution_bar / model_contribution: unchanged
    (still based on argmax category and ev).

TP-mode variant: tailored for True-Positive oligodendroglioma (1p/19q codeleted)
WSIs. Only the four oligo concepts (k=0..3: round nuclei, fried-egg cytoplasm,
chicken-wire vessels, microcalcification) are rendered; the astro concept (k=4)
and the "others" map are skipped. The verification figure
(``tp_oligo_verification.pdf``) renders one row per selected oligo concept,
labelled in user-input sequential order — real ranks never appear in any PDF
text element and are persisted only in the sidecar JSON / coords.csv.

Loads a trained MorphoPath checkpoint (default: the no-grade model,
``lambda_grade=0``) and the bundled CONCH anchors (``src/conch_{loc,score}.pt``);
all paths are overridable via CLI flags. Output is written under ``visual/``.

Figure details:
  - PDF-only output for the main figures (no PNG, no SVG).
  - Arial / Helvetica font enforcement (PDF type-42, editable text).
  - Tableau-aligned palette: #E64B35 (codeleted, coral) / #4DBBD5 (non-codeleted,
    cerulean) / #91D1C2 (others, mint).
  - Per-concept oligo maps use a white -> coral MorphoPath cmap; astro map (only
    rendered in non-TP mode) uses a matching white -> cerulean cmap; others map
    (only rendered in non-TP mode) uses an optional white -> mint ramp.
  - 1 mm white scale bar with thin black outline on concept_label_heatmap only.
  - Top-N patch dump block (top20/<category>/rank*.png + coords.csv) reads slide
    pyramid level 1 (with fallback to level 0) and writes optimised PNG crops.
  - Batch driver via --batch reads visual/slide_vis/folder_names.xlsx (rows
    where column 1 == 1) and runs the pipeline per WSI sequentially.

CLI args, defaults, and types match ``viz_slide.py`` except:
  - --vis_level default 3 -> 1 (sharper background thumbnail)
  - --batch flag added (single-WSI invocation otherwise unchanged)
  - --manual_picks accepts both per-concept syntax
    ("nuclear:1,2,3 cytoplasm:1,2 vascular:1 calcification:1") and a simple
    comma-separated list ("1,2,5") that defaults to the nuclear (k=0) concept.
"""
import os, sys, csv, argparse, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image as _PIL_Image
try:
    from skimage import color as _sk_color
    _SK_OK = True
except Exception:
    _SK_OK = False


class _ReinhardNormalizer:
    """LAB-space Reinhard stain normalization (Macenko-style). Loads reference
    image once at init, then __call__(PIL.Image) → normalized PIL.Image.
    Mirrors gigapath_feature_extract.py:ReinhardNormalizer."""
    def __init__(self, ref_img_path):
        if not _SK_OK:
            raise RuntimeError("skimage required for ReinhardNormalizer")
        tgt = _PIL_Image.open(ref_img_path).convert("RGB")
        arr = np.asarray(tgt).astype(np.float32) / 255.0
        lab = _sk_color.rgb2lab(arr)
        self.tgt_mean = np.array([lab[:,:,i].mean() for i in range(3)])
        self.tgt_std  = np.array([lab[:,:,i].std()  for i in range(3)])

    def __call__(self, pil_img):
        arr = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
        lab = _sk_color.rgb2lab(arr)
        for i in range(3):
            mu = lab[:,:,i].mean(); sigma = lab[:,:,i].std()
            lab[:,:,i] = ((lab[:,:,i] - mu) / (sigma + 1e-8)) * self.tgt_std[i] + self.tgt_mean[i]
        rgb = _sk_color.lab2rgb(lab) * 255.0
        return _PIL_Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


_REINHARD = None   # module-level, populated in run_one() from --reinhard_ref

# --- Font / rcParams enforcement BEFORE any figure creation -----------------
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'legend.frameon': True,
})
from matplotlib import font_manager as _fm
_resolved_font = Path(_fm.findfont(mpl.rcParams['font.sans-serif'][0],
                                   fallback_to_default=True)).name
if 'arial' not in _resolved_font.lower() and 'helvetica' not in _resolved_font.lower():
    raise RuntimeError(
        f"findfont fell back to '{_resolved_font}'; Arial / Helvetica required."
    )

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# Local src — model + viz helpers
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent))  # repo root -> main src/
from src.morphopath import MorphoPath
from src.visualize import compute_patch_spacing, _ms as _ms_helper, _vis_coords

# ---------------------------------------------------------------------------
# Default paths (override via CLI). These point at the repo's canonical locations.
# ---------------------------------------------------------------------------
REPO = HERE.parent                          # …/1p19q
DEFAULTS = dict(
    ckpt      = REPO / "results/morphopath/best_morphopath_41_seed42.pt",
    loc_pt    = REPO / "src/conch_loc.pt",
    score_pt  = REPO / "src/conch_score.pt",
    feat_dir  = REPO / "1p19q_data/HE_WSI_BTH_512",
    raw_dir   = Path("/Volumes/Expansion/mq-tt/1p19q/test_tt"),
    label_xlsx= REPO / "1p19q_data/BTH_List.xlsx",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("wsi_id", nargs="?", default=None,
                   help="WSI ID matching the feature/raw directory subfolder "
                        "(omit when --batch is used)")
    p.add_argument("--label", default="TP", help="cohort tag appended to out_dir name (TP/TN/FP/FN)")
    p.add_argument("--ckpt", default=str(DEFAULTS["ckpt"]))
    p.add_argument("--loc_pt", default=str(DEFAULTS["loc_pt"]))
    p.add_argument("--score_pt", default=str(DEFAULTS["score_pt"]))
    p.add_argument("--feat_dir", default=str(DEFAULTS["feat_dir"]))
    p.add_argument("--raw_dir", default=str(DEFAULTS["raw_dir"]))
    p.add_argument("--label_xlsx", default=str(DEFAULTS["label_xlsx"]))
    p.add_argument("--out_dir", default=None,
                   help="default: visual/vis_filtered/<WSI>")
    p.add_argument("--out_root", default=None,
                   help="parent dir for batch mode; each WSI placed under <out_root>/<wsi_id>")
    p.add_argument("--sal_pct", type=float, default=None,
                   help="per-slide saliency percentile cutoff; default 0.0 means quantile disabled and sal_floor takes over (was 0.10 quantile)")
    p.add_argument("--sal_floor", type=float, default=0.01,
                   help="absolute saliency floor; effective cutoff = max(quantile(sal_pct), sal_floor). Default 0.01 = absolute threshold; varies others%% naturally across slides.")
    p.add_argument("--pick_sal_upper_pct", type=float, default=99.0,
                   help="percentile cap on saliency for the top-N selection pool ONLY. "
                        "Patches with saliency > p<pick_sal_upper_pct> are excluded from "
                        "greedy diversification, rank_maps, and save_top picks (the 4 "
                        "concept heatmaps and model_contribution remain unchanged). "
                        "Default 99 drops the top 1%% high-saliency tail to suppress "
                        "artifact/confounder patches. Set 100 to disable.")
    p.add_argument("--pick_sal_upper_abs", type=float, default=None,
                   help="ABSOLUTE saliency upper cap; if set, overrides pick_sal_upper_pct. "
                        "Patches with saliency >= pick_sal_upper_abs are excluded from "
                        "top-N selection. Use 0.97 to drop the obvious confounder tail.")
    p.add_argument("--relevance_pct", type=float, default=50.0,
                   help="Percentile of attn_total (over diag patches) used as the "
                        "relevance gate for the heatmap display_mask. Default 50 = "
                        "only colour the top 50%% of diag patches by attn_total; "
                        "set 0 to disable (display_mask = is_diag).")
    p.add_argument("--skip_morph_top", action="store_true",
                   help="Skip the parallel top20_morph/ dump (pure morph_attn ranking). "
                        "Only top20/ (attn-ranked) is generated.")
    p.add_argument("--reinhard_ref", default=None,
                   help="Reference image path for LAB-space Reinhard stain "
                        "normalization applied to top-N PNG crops. If unset, "
                        "PNG is saved raw (current default).")
    p.add_argument("--vis_level", type=int, default=1)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--rank_by", default="attn", choices=["attn", "ev"],
                   help="ranking key for round-robin top-N: 'attn' (attn_np) "
                        "or 'ev' (attn × concept_score × scale)")
    p.add_argument("--input_dim", type=int, default=1024)
    p.add_argument("--n_concepts", type=int, default=6)
    p.add_argument("--n_diagnostic", type=int, default=5)
    p.add_argument("--n_oligo", type=int, default=4)
    p.add_argument("--target_patch_um", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--batch", action="store_true",
                   help="run all WSIs flagged in folder_names.xlsx (col 1 == 1)")
    p.add_argument("--manual_picks", default="",
                   help="Manual picks for the TP verification figure. Two formats: "
                        "(a) per-concept 'nuclear:1,2,3 cytoplasm:1,2 vascular:1 calcification:1'; "
                        "(b) simple '1,2,5' which defaults to the nuclear (k=0) concept.")
    p.add_argument("--xlsx", default=str(HERE / "folder_names.xlsx"),
                   help="path to folder_names.xlsx (used with --batch)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Labels and palette (npj DM / MorphoPath family)
# ---------------------------------------------------------------------------
CONCEPT_NAMES = ["oligo_nuclear", "oligo_cytoplasm", "oligo_vascular",
                 "oligo_calcification", "astro_nuclear"]
MORPH_SHORTS = ["round nuclei", "fried-egg cytoplasm",
                "chicken-wire vessels", "microcalcification",
                "pleomorphic nuclei"]

# Per-oligo-concept human-readable short names (k=0..3) used in the TP
# verification figure row sub-titles.
OLIGO_SHORTS = {
    0: "round nuclei",
    1: "fried-egg cytoplasm",
    2: "chicken-wire vessels",
    3: "microcalcification",
    4: "pleomorphic nuclei (astro)",
}

# Alias-table for parsing --manual_picks in per-concept form.
# All aliases lower-case; map to integer concept index k in {0,1,2,3}.
_OLIGO_ALIASES = {
    "nuclear": 0, "oligo_nuclear": 0, "k0": 0,
    "cytoplasm": 1, "oligo_cytoplasm": 1, "k1": 1,
    "vascular": 2, "oligo_vascular": 2, "k2": 2,
    "calcification": 3, "oligo_calcification": 3, "k3": 3,
    # TN-mode aliases for astro (k=4)
    "astro": 4, "astro_nuclear": 4, "k4": 4,
}

# Canonical alias used as JSON/meta key for each concept (oligo + astro).
_OLIGO_ALIAS_CANON = {0: "nuclear", 1: "cytoplasm", 2: "vascular",
                      3: "calcification", 4: "astro"}
_OLIGO_DIR_NAMES = {0: "oligo_nuclear", 1: "oligo_cytoplasm",
                    2: "oligo_vascular", 3: "oligo_calcification",
                    4: "astro_nuclear"}

# Nature/npg bright professional palette (Nature Publishing Group inspired).
CODELETED     = '#E64B35'   # vivid coral
NON_CODELETED = '#4DBBD5'   # bright cerulean
OTHERS        = '#91D1C2'   # mint

GROUP3_COLORS    = [CODELETED, NON_CODELETED, OTHERS]
GROUP3_NAMES_DISP = ["1p/19q codeleted", "1p/19q non-codeleted", "others"]

# Per-concept evidence colormaps (k-anchored) - Nature/npg bright family.
MORPHOPATH_CMAP = LinearSegmentedColormap.from_list(
    "morpho_red",
    ['#FFFFFF', '#FCE0DC', '#F58F77', '#E64B35'],
    N=256,
)
# Alias retained for any downstream references.
CMAP_OLIGO = MORPHOPATH_CMAP

# Astro evidence map: white -> bright cerulean.
ASTRO_CMAP = LinearSegmentedColormap.from_list(
    "morpho_blue",
    ['#FFFFFF', '#D4EEF7', '#7FCFE3', '#4DBBD5'],
    N=256,
)
CMAP_ASTRO = ASTRO_CMAP

# Others map: white -> mint.
OTHERS_CMAP = LinearSegmentedColormap.from_list(
    "morpho_green",
    ['#FFFFFF', '#E1F2EC', '#B3DDD0', '#91D1C2'],
    N=256,
)
CMAP_OTHERS = OTHERS_CMAP

# Figure sizing for paper inclusion (npj DM single-column = 89 mm).
MM_PER_INCH = 25.4
TARGET_W_IN = 89.0 / MM_PER_INCH      # ≈ 3.504 in


def _save_pdf(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches=None, pad_inches=0.10,
                dpi=300)


def _parse_tp_manual_picks(raw):
    """Parse the --manual_picks argument for TP mode.

    Returns a dict mapping concept index k in {0,1,2,3} to an ordered list of
    integer ranks (1-indexed, user-input order, duplicates preserved). Returns
    an empty dict if the string is empty / malformed.

    Two accepted formats:
      (a) per-concept: "nuclear:1,2,3 cytoplasm:1,2 vascular:1 calcification:1"
          (space-separated entries; aliases per ``_OLIGO_ALIASES``).
      (b) simple: "1,2,5" — defaults to nuclear (k=0).
    """
    text = (raw or "").strip()
    if not text:
        return {}
    picks_by_k = {}
    # Detect per-concept form by presence of a ':' anywhere.
    if ":" in text:
        for entry in text.split():
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            concept_tok, ranks_tok = entry.split(":", 1)
            key = concept_tok.strip().lower()
            if key not in _OLIGO_ALIASES:
                print(f"[tp-verify] unknown concept alias: {concept_tok!r}; skipping entry")
                continue
            k = _OLIGO_ALIASES[key]
            ranks = []
            for tok in ranks_tok.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    ranks.append(int(tok))
                except ValueError:
                    print(f"[tp-verify] skip non-integer pick {tok!r} for concept k={k}")
            if ranks:
                picks_by_k.setdefault(k, []).extend(ranks)
    else:
        # Simple comma-list: defaults to nuclear (k=0).
        ranks = []
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ranks.append(int(tok))
            except ValueError:
                print(f"[tp-verify] skip non-integer pick: {tok!r}")
        if ranks:
            picks_by_k[0] = ranks
    return picks_by_k


def _render_tp_verification(args, out_dir, top_dir, thumbnail, cv_x, cv_y, half,
                            sw_px, sh_px, ev_np, category, ms,
                            patch_spacing_px, downsample, intensity_np=None,
                            display_mask=None, vabs_symmetric=None,
                            diverging=False):
    """Generate tp_oligo_verification.pdf: one row per selected oligo concept.

    For each selected concept k in {0,1,2,3}, the row contains:
      - Left: that concept's evidence heatmap (CMAP_OLIGO white→#E64B35).
      - Right: patches from top20/<concept>/rank{XX}.png at the user-specified
        ranks, stacked horizontally.

    Patch labels are SEQUENTIAL 1-based indices in the user-provided order,
    NOT the real rank numbers. The real rank mapping is persisted to
    ``tp_oligo_verification_meta.json``. Real ranks remain available in
    ``top20/<concept>/coords.csv``.
    """
    import json as _json
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle as _Rect
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    picks_by_k = _parse_tp_manual_picks(args.manual_picks)
    if not picks_by_k:
        print("[tp-verify] no valid manual picks; skipping")
        return

    # Stable ordering of selected concepts: k = 0, 1, 2, 3 (skip absent).
    selected_ks = [k for k in (0, 1, 2, 3, 4) if picks_by_k.get(k)]
    if not selected_ks:
        print("[tp-verify] no oligo concepts selected after parsing; skipping")
        return

    # Load coords.csv per selected concept so we can resolve per-rank ev.
    coords_by_k = {}
    for k in selected_ks:
        concept_dir_name = _OLIGO_DIR_NAMES[k]
        coords_csv = top_dir / concept_dir_name / "coords.csv"
        if not coords_csv.exists():
            print(f"[tp-verify] coords.csv missing at {coords_csv}; "
                  f"skipping concept k={k}")
            continue
        rank_rows = {}
        with open(coords_csv, "r", newline="") as fp:
            rd = csv.DictReader(fp)
            for row in rd:
                try:
                    rank_rows[int(row["rank"])] = row
                except (ValueError, KeyError):
                    continue
        coords_by_k[k] = rank_rows
    selected_ks = [k for k in selected_ks if k in coords_by_k]
    if not selected_ks:
        print("[tp-verify] no concepts have coords.csv; skipping")
        return

    # --- v11.10 — Dump picked patches to a dedicated subfolder ---------------
    # For each manual pick, save BOTH raw and Reinhard-normalized PNG to
    # `manual_picks/`. Filename: <concept>_disp{i}_rank{pk}_idx{patch_idx}_{raw|norm}.png
    picks_dir = out_dir / "manual_picks"
    picks_dir.mkdir(exist_ok=True)
    for _old in list(picks_dir.glob("*.png")):
        try: _old.unlink()
        except Exception: pass
    _svs = None
    for _ext in [".svs", ".ndpi", ".tif", ".tiff"]:
        _bare = args.wsi_id
        for _suf in ("_TP","_TN","_FP","_FN"):
            if _bare.endswith(_suf): _bare = _bare[:-len(_suf)]; break
        _cand = Path(args.raw_dir) / f"{_bare}{_ext}"
        if _cand.exists(): _svs = _cand; break
    _lvl_size_l0 = patch_spacing_px
    try:
        import openslide as _osl
        _slide = _osl.OpenSlide(str(_svs)) if _svs and _svs.exists() else None
    except Exception:
        _slide = None
    if _slide is not None:
        _lvl = 1 if _slide.level_count >= 2 else 0
        _ds = _slide.level_downsamples[_lvl]
        _sz = max(1, int(round(patch_spacing_px / _ds)))
        for k in selected_ks:
            concept = _OLIGO_DIR_NAMES[k]
            for col_j, pk in enumerate(picks_by_k[k], 1):
                row = coords_by_k[k].get(pk)
                if not row: continue
                try:
                    x0 = int(row["coord_x_pixel"]); y0 = int(row["coord_y_pixel"])
                    pid = row.get("patch_idx", "")
                except (KeyError, ValueError): continue
                try:
                    raw_img = _slide.read_region((x0, y0), _lvl, (_sz, _sz)).convert("RGB")
                    raw_img.save(picks_dir / f"{concept}_disp{col_j}_rank{pk}_idx{pid}_raw.png",
                                 "PNG", optimize=True, compress_level=6)
                    if _REINHARD is not None:
                        norm_img = _REINHARD(raw_img)
                        norm_img.save(picks_dir / f"{concept}_disp{col_j}_rank{pk}_idx{pid}_norm.png",
                                      "PNG", optimize=True, compress_level=6)
                except Exception as e:
                    print(f"[tp-verify-picks] {concept} disp{col_j} rank{pk}: {e}")
        _slide.close()
        n_raw = len(list(picks_dir.glob("*_raw.png")))
        n_norm = len(list(picks_dir.glob("*_norm.png")))
        print(f"[tp-verify-picks] saved {n_raw} raw + {n_norm} norm patches → {picks_dir.relative_to(out_dir)}/")
    else:
        print(f"[tp-verify-picks] slide not found at {_svs}; skipping patch dump")

    # --- Figure layout: one row per concept; left heatmap + right patch row -----
    n_rows = len(selected_ks)
    # Width ~ 200 mm wide so a horizontal patch strip up to ~6 patches fits.
    fig_w_in = 200.0 / MM_PER_INCH
    row_h_mm = 70.0
    fig_h_in = (row_h_mm * n_rows + 14.0) / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    outer = fig.add_gridspec(n_rows, 2,
                             width_ratios=[1.4, 2.4],
                             wspace=0.10, hspace=0.30,
                             left=0.04, right=0.985,
                             top=0.90, bottom=0.04)

    for row_i, k in enumerate(selected_ks):
        picks = list(picks_by_k[k])
        rank_rows = coords_by_k[k]

        # ---- Left: per-concept evidence heatmap (oligo cmap) ------------------
        ax_hm = fig.add_subplot(outer[row_i, 0])
        if thumbnail is not None:
            ax_hm.imshow(thumbnail, extent=[0, sw_px, sh_px, 0], aspect="equal",
                         alpha=0.95, interpolation='lanczos')
        ax_hm.set_xlim(0, sw_px); ax_hm.set_ylim(sh_px, 0)
        ax_hm.set_aspect("equal"); ax_hm.axis("off"); ax_hm.set_facecolor("white")

        _intensity = intensity_np if intensity_np is not None else ev_np
        int_k = _intensity[:, k]
        # v11.9 — diverging mode (morph_zcross): paint ALL display_mask patches
        # with RdBu_r symmetric ±vabs (matches concept_*_oligo_*.pdf heatmap).
        # Sequential mode (legacy attn): only category==k argmax winners with
        # MORPHOPATH_CMAP white→red, vmax=p99.
        if diverging and display_mask is not None:
            disp = display_mask; ndisp = ~display_mask
            if ndisp.any():
                ax_hm.scatter(cv_x[ndisp] + half, cv_y[ndisp] + half,
                              c="lightgray", s=ms * 0.4, alpha=0.10,
                              edgecolors="none", marker="s")
            if disp.any():
                vals = int_k[disp]
                vabs = vabs_symmetric if vabs_symmetric is not None else \
                       max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)), 1e-12)
                sc = ax_hm.scatter(cv_x[disp] + half, cv_y[disp] + half,
                                   c=vals, cmap="RdBu_r",
                                   vmin=-vabs, vmax=+vabs,
                                   s=ms, alpha=0.85, edgecolors="none", marker="s")
                cb = plt.colorbar(sc, ax=ax_hm, shrink=0.25, pad=0.015, aspect=25)
                cb.ax.tick_params(labelsize=6, width=0.5, length=2)
                for _t in cb.ax.get_yticklabels(): _t.set_fontfamily('sans-serif')
                cb.set_label("morph z (across slide)", fontsize=8, rotation=90, labelpad=6)
                cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor('black')
        else:
            is_k = (category == k)
            if (~is_k).any():
                ax_hm.scatter(cv_x[~is_k] + half, cv_y[~is_k] + half,
                              c="lightgray", s=ms * 0.4, alpha=0.10,
                              edgecolors="none", marker="s")
            if is_k.any():
                vals = int_k[is_k]
                if vals.max() > 0:
                    vmax = max(np.percentile(vals, 99), 1e-12)
                    sc = ax_hm.scatter(cv_x[is_k] + half, cv_y[is_k] + half,
                                       c=vals, cmap=MORPHOPATH_CMAP,
                                       vmin=0, vmax=vmax,
                                       s=ms, alpha=0.85, edgecolors="none", marker="s")
                    cb = plt.colorbar(sc, ax=ax_hm, shrink=0.25, pad=0.015, aspect=25)
                    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
                    for _t in cb.ax.get_yticklabels(): _t.set_fontfamily('sans-serif')
                    cb.set_label("Attn", fontsize=8, rotation=90, labelpad=6)
                    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor('black')

        # 1 mm scale bar per heatmap.
        try:
            px_per_um_l0 = patch_spacing_px / float(args.target_patch_um)
            bar_len_vis = (1000.0 * px_per_um_l0) / float(downsample if downsample else 1.0)
            bar_h = max(sh_px * 0.006, 2.0)
            margin_x = sw_px * 0.04
            margin_y = sh_px * 0.06
            x0_bar = sw_px - margin_x - bar_len_vis
            y0_bar = sh_px - margin_y - bar_h
            ax_hm.add_patch(_Rect((x0_bar, y0_bar), bar_len_vis, bar_h,
                                  facecolor="white", edgecolor="black",
                                  linewidth=0.6, zorder=10))
            ax_hm.text(x0_bar + bar_len_vis / 2, y0_bar + bar_h + sh_px * 0.018,
                       "1 mm", ha="center", va="top", fontsize=8, color="black",
                       zorder=11)
        except Exception as e:
            print(f"[tp-verify scalebar] skipped: {e}")

        # Highlight picked patches with thin black square outlines + sequential idx.
        for display_idx, pk in enumerate(picks, start=1):
            row = rank_rows.get(pk)
            if row is None:
                continue
            try:
                idx = int(row["patch_idx"])
            except (KeyError, ValueError):
                continue
            x_c = float(cv_x[idx]); y_c = float(cv_y[idx])
            side = max(half * 2.2, 4.0)
            ax_hm.add_patch(_Rect((x_c + half - side / 2, y_c + half - side / 2),
                                  side, side, fill=False,
                                  edgecolor="black", linewidth=0.5, alpha=1.0,
                                  zorder=20))
            ax_hm.text(x_c + half + side / 2 + sw_px * 0.005,
                       y_c + half - side / 2,
                       f"{display_idx}", fontsize=7, color="black",
                       va="top", ha="left", zorder=21)

        # Row sub-title (human-readable concept name) above the heatmap.
        ax_hm.set_title(OLIGO_SHORTS[k], fontsize=9, pad=4,
                        fontfamily='sans-serif')

        # ---- Right: horizontally stacked patch thumbnails ---------------------
        patch_dir = top_dir / _OLIGO_DIR_NAMES[k]
        n_patches = len(picks)
        inner = GridSpecFromSubplotSpec(1, n_patches,
                                        subplot_spec=outer[row_i, 1],
                                        wspace=0.10)
        for col_j, pk in enumerate(picks):
            display_idx = col_j + 1
            ax_p = fig.add_subplot(inner[0, col_j])
            ax_p.set_xticks([]); ax_p.set_yticks([])
            for spine in ax_p.spines.values():
                spine.set_linewidth(1.0); spine.set_color("black"); spine.set_alpha(1.0)
            png_path = patch_dir / f"rank{pk:02d}.png"
            if png_path.exists():
                try:
                    img = mpimg.imread(str(png_path))
                    ax_p.imshow(img, aspect="equal", alpha=1.0)
                except Exception as e:
                    ax_p.text(0.5, 0.5, f"load fail\n{e}",
                              ha="center", va="center", transform=ax_p.transAxes,
                              fontsize=6)
            else:
                ax_p.text(0.5, 0.5, "missing patch",
                          ha="center", va="center", transform=ax_p.transAxes,
                          fontsize=6)
            row = rank_rows.get(pk, {})
            attn_val = row.get("attn_value", "")
            try:
                attn_str = f"{float(attn_val):.3f}"
            except (TypeError, ValueError):
                attn_str = str(attn_val)
            # Sequential display label ONLY ("1", "2", ...) — 12 pt bold black
            # on white bbox; real rank never appears in PDF text.
            ax_p.text(0.04, 0.96, f"{display_idx}",
                      transform=ax_p.transAxes,
                      ha="left", va="top",
                      fontsize=12, fontweight='bold', color='black',
                      fontfamily='sans-serif',
                      bbox=dict(boxstyle='square,pad=0.20',
                                facecolor='white', edgecolor='none', alpha=1.0))
            # Optional subtle grey attn value below the label inside the patch.
            ax_p.text(0.04, 0.04, f"attn={attn_str}",
                      transform=ax_p.transAxes,
                      ha="left", va="bottom",
                      fontsize=6, fontweight='normal', color='#777777',
                      fontfamily='sans-serif')

    # Top title.
    fig.suptitle(f"{args.wsi_id}  \u2022  "
                 f"Oligodendroglioma, IDH-mutant, 1p/19q codeleted",
                 fontsize=10, y=0.985, fontfamily='sans-serif')

    out_pdf = out_dir / "tp_oligo_verification.pdf"
    fig.savefig(out_pdf, bbox_inches=None, pad_inches=0.10, dpi=300)
    plt.close(fig)
    actual_ranks = {_OLIGO_ALIAS_CANON[k]: list(picks_by_k[k]) for k in selected_ks}
    print(f"[tp-verify] wrote {out_pdf.name} ({out_pdf.stat().st_size} B)  "
          f"actual_ranks={actual_ranks}")

    # --- Sidecar JSON: real-rank mapping (NOT in PDF) -------------------------
    meta = {
        "wsi_id": args.wsi_id,
        "manual_picks_actual_ranks": {
            _OLIGO_ALIAS_CANON[k]: list(picks_by_k[k]) for k in selected_ks
        },
        "display_labels": {
            _OLIGO_ALIAS_CANON[k]: list(range(1, len(picks_by_k[k]) + 1))
            for k in selected_ks
        },
        "mapping": {
            _OLIGO_ALIAS_CANON[k]: {str(i + 1): pk
                                    for i, pk in enumerate(picks_by_k[k])}
            for k in selected_ks
        },
    }
    meta_path = out_dir / "tp_oligo_verification_meta.json"
    with open(meta_path, "w") as fp:
        _json.dump(meta, fp, indent=4)
    print(f"[tp-verify] wrote {meta_path.name}  mapping={meta['mapping']}")


def run_one(args):
    """Run the full per-WSI visualisation pipeline.

    Expects args.wsi_id and args.label to be set. Returns the output dir.
    """
    # --- Reinhard stain normalizer (lazy init from CLI ref path) ------------
    global _REINHARD
    _REINHARD = None
    if getattr(args, "reinhard_ref", None):
        _REINHARD = _ReinhardNormalizer(args.reinhard_ref)
        print(f"[reinhard] enabled, ref = {args.reinhard_ref}")
    # --- Resolve TP-mode defaults --------------------------------------------
    is_tp_mode = (args.label.upper() == "TP")
    # sal_pct default: 0.0 (quantile disabled). The effective cutoff becomes
    # max(min(sal), sal_floor) = sal_floor, which is 0.01 by default — an
    # absolute saliency threshold that lets others% vary naturally per slide.
    # None means user did not supply --sal_pct on the CLI.
    if args.sal_pct is None:
        args.sal_pct = 0.0
        if is_tp_mode:
            print(f"[TP mode] using absolute saliency floor "
                  f"sal_pct=0.0, sal_floor={args.sal_floor}")

    device = torch.device(args.device) if args.device else \
             torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  wsi={args.wsi_id}  sal_pct={args.sal_pct}  "
          f"vis_level={args.vis_level}", flush=True)
    print(f"[font ] resolved sans-serif -> {_resolved_font}", flush=True)

    # --- Output directory ----------------------------------------------------
    # TP mode without --out_dir override -> visual/tp_verification/<WSI>/.
    # The wsi_id may carry a "_TP" suffix from the CLI (e.g. "126916_TP");
    # we keep it as given in the path so users can grep for the exact ID.
    _is_tn_outdir = (args.label.upper() == "TN")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif is_tp_mode:
        out_dir = REPO / "visual" / "tp_verification" / args.wsi_id
        print(f"[TP mode] out_dir auto-routed to {out_dir}")
    elif _is_tn_outdir:
        out_dir = REPO / "visual" / "tn_verification_floor01" / f"{args.wsi_id}_TN"
        print(f"[TN mode] out_dir auto-routed to {out_dir}")
    else:
        out_dir = REPO / "visual" / "vis_filtered_nograde" / args.wsi_id
    out_dir.mkdir(parents=True, exist_ok=True)
    top_dir = out_dir / "top20"
    top_dir.mkdir(exist_ok=True)

    # --- Load model ----------------------------------------------------------
    N_OLIGO  = args.n_oligo
    N_DIAG   = args.n_diagnostic
    model = MorphoPath(
        tau_init=5.0, n_diagnostic=N_DIAG, use_normal_anchor=True,
        input_dim=args.input_dim, proj_dim=512, attn_dim=256,
        n_concepts=args.n_concepts, n_oligo=N_OLIGO,
        conch_loc_path=str(args.loc_pt), conch_score_path=str(args.score_pt),
        dropout=0.1, tau_score=5.0,
        lambda_loc_bias=0.01, sign_constraint=True,
        attn_mode="residual", tau_loc=1.0,
    ).to(device)
    model.load_state_dict(torch.load(str(args.ckpt), map_location=device, weights_only=True))
    model.eval()

    # --- Load features -------------------------------------------------------
    # Some callers pass wsi_id with a trailing label suffix (e.g. 126916_TP).
    # Strip a known label suffix for filesystem lookups; preserve the original
    # for downstream display/output paths.
    _bare_wsi = args.wsi_id
    for _suf in ("_TP", "_TN", "_FP", "_FN"):
        if _bare_wsi.endswith(_suf):
            _bare_wsi = _bare_wsi[: -len(_suf)]
            break
    folder = Path(args.feat_dir) / _bare_wsi
    if not folder.exists():
        # Fallback to the original (un-stripped) form.
        folder = Path(args.feat_dir) / args.wsi_id
    h5_path = next(p for p in folder.glob("*_features.h5") if not p.name.startswith("._"))
    with h5py.File(h5_path, "r") as f:
        features = torch.from_numpy(f["features"][:]).float().to(device)
        coords = torch.from_numpy(f["coords"][:])
    coords_np = coords.numpy().astype(float)
    N = len(coords_np)

    # --- Forward + extract intermediate tensors -------------------------------
    with torch.no_grad():
        logit, _, concept_scores, attn_weights, _ = model(features, coords.to(device).float())
        prob = float(torch.sigmoid(logit).item())
        saliency = model.get_saliency(features).cpu().numpy()
        h = model.projection(features); h_norm = F.normalize(h, dim=-1)
        w_loc_norm = F.normalize(model.W_loc[:N_DIAG], dim=-1)
        loc_sim = h_norm @ w_loc_norm.T
        morph_attn = F.softmax(loc_sim / model.tau_loc, dim=0).cpu().numpy()
    cs_np   = concept_scores.cpu().numpy()
    attn_np = attn_weights.cpu().numpy()
    concept_scale = np.ones(N_DIAG, dtype=float)
    concept_scale[N_OLIGO:] = N_OLIGO / (N_DIAG - N_OLIGO)
    ev_np = attn_np * cs_np * concept_scale[np.newaxis, :]

    # OVERLAP MODE intensity / ranking signal: per-patch concept SHARE.
    # share[i, k] = attn[i, k] / sum_j attn[i, j]  — LINEAR proportion across
    # the 5 diagnostic concepts (sums to 1 over k per patch). We use linear
    # normalisation, NOT softmax, because attn values are ~1e-3 (softmax of
    # tiny inputs saturates to uniform 1/K and the resulting "share" becomes
    # noise-driven). Linear normalisation preserves the multiplicative ratios
    # between concepts, which is the signal we want.
    # Uses only forward variables (attn_np from the model's forward pass);
    # the cross-concept normalisation is the same intuition as the old
    # argmax(attn, axis=1) but kept as a soft (continuous) preference instead
    # of a hard winner.
    _attn_diag = attn_np[:, :N_DIAG]
    _denom = _attn_diag.sum(axis=1, keepdims=True)
    _denom = np.where(_denom > 1e-12, _denom, 1.0)   # guard against /0
    concept_share = _attn_diag / _denom

    # OVERLAP_SAL: per-patch concept SPECIFICITY = attn_np minus 4-oligo mean.
    # attn_specific[i, k] = attn_np[i, k] - mean_j(attn_np[i, :N_OLIGO])
    # Positive: this patch attends to k MORE than to the 4-oligo average.
    # Negative: this patch attends to k LESS than to the 4-oligo average.
    # Row-wise zero-sum across 4 oligo concepts.
    # Removes saliency's shared shape (cross-concept corr 0.999 in attn_np
    # gets reduced to actual concept-specific deviation).
    _attn_4oligo = attn_np[:, :N_OLIGO]
    attn_specific = _attn_4oligo - _attn_4oligo.mean(axis=1, keepdims=True)
    # OVERLAP_SAL v11.3: also compute per-patch z-score (mean-subtracted /
    # std). Heatmap switched from attn_specific to attn_z so that low-attn
    # patches with the SAME concept preference signature get the same color
    # intensity as high-attn ones (std-normalized → comparable across patches).
    _attn_std = _attn_4oligo.std(axis=1, keepdims=True)
    attn_z = attn_specific / (_attn_std + 1e-12)

    sal_cut = max(float(np.quantile(saliency, args.sal_pct)), float(args.sal_floor))
    # OVERLAP_SAL v11.2: saliency upper cap applied to is_diag itself, so it
    # propagates to BOTH heatmaps AND top-N selection. Capped-out patches show
    # as light-grey background in all maps (same as below sal_cut). The model's
    # slide_profile / prediction is computed earlier and unchanged.
    if args.pick_sal_upper_abs is not None:
        sal_upper = float(args.pick_sal_upper_abs)
        _upper_src = f"abs={sal_upper:.4f}"
    elif args.pick_sal_upper_pct < 100:
        sal_upper = float(np.quantile(saliency, args.pick_sal_upper_pct / 100.0))
        _upper_src = f"p{args.pick_sal_upper_pct}={sal_upper:.4f}"
    else:
        sal_upper = float('inf')
        _upper_src = "disabled"
    is_diag_raw = saliency > sal_cut
    is_diag = is_diag_raw & (saliency < sal_upper)
    is_pick = is_diag       # kept as alias for backward compat in pick paths
    _n_dropped = int(is_diag_raw.sum() - is_diag.sum())

    # OVERLAP_MORPHZCROSS v11.6: heatmap = per-concept z of MORPH_ATTN across
    # patches (saliency/score-free → real concept differentiation, r~0.41).
    # Only "model-relevant" patches are coloured:
    #   display_mask = is_diag AND attn_total >= p50_diag(attn_total)
    _morph_4oligo = morph_attn[:, :N_OLIGO]
    if is_diag.any():
        _m_diag = _morph_4oligo[is_diag]
        _mu = _m_diag.mean(axis=0, keepdims=True)
        _sd = _m_diag.std(axis=0, keepdims=True)
        morph_zcross = (_morph_4oligo - _mu) / (_sd + 1e-12)
    else:
        morph_zcross = np.zeros_like(_morph_4oligo)
    # TN-mode: also compute morph_zcross_astro (k=N_OLIGO=4, single column).
    if is_diag.any():
        _ast = morph_attn[is_diag, N_OLIGO]
        morph_zcross_astro = (morph_attn[:, N_OLIGO] - _ast.mean()) / (_ast.std() + 1e-12)
    else:
        morph_zcross_astro = np.zeros(morph_attn.shape[0])
    _attn_total = attn_np[:, :N_OLIGO].sum(axis=1)
    if args.relevance_pct > 0 and is_diag.any():
        _attn_cut = float(np.percentile(_attn_total[is_diag], args.relevance_pct))
        relevance_mask = _attn_total >= _attn_cut
        display_mask = is_diag & relevance_mask
        _cut_str = f"attn_total p{args.relevance_pct} over diag = {_attn_cut:.4e}"
    else:
        display_mask = is_diag.copy()
        _cut_str = "disabled (display_mask = is_diag)"
    print(f"[morphzcross] {_cut_str}; "
          f"display_mask={display_mask.sum()} of diag={is_diag.sum()} "
          f"({100*display_mask.sum()/max(1,is_diag.sum()):.1f}%)")

    assignment = morph_attn.argmax(axis=1)
    category = np.where(is_diag, assignment, 5)     # 0..4 = concept, 5 = others
    group3 = np.where(category < N_OLIGO, 0,
             np.where(category == N_OLIGO, 1, 2))   # 0 = 1p/19q codel, 1 = non-codel, 2 = others

    print(f"[stats] N={N}  prob={prob:.3f}  sal mean={saliency.mean():.3f}  "
          f"sal_cut={sal_cut:.3f} (pct={args.sal_pct} floor={args.sal_floor})  "
          f"diagnostic_raw={is_diag_raw.mean()*100:.1f}%  "
          f"diagnostic_final={is_diag.mean()*100:.1f}%")
    print(f"[pick_filter] sal_upper({_upper_src}) — applied to BOTH heatmap AND top-N; "
          f"dropped {_n_dropped} patches from diagnostic pool")
    for c in range(6):
        cnt = int((category == c).sum())
        name = CONCEPT_NAMES[c] if c < 5 else "others"
        print(f"  {name:<22}  {cnt:>5}  ({cnt/N*100:5.2f}%)")

    # --- Thumbnail / scaling --------------------------------------------------
    try:
        import openslide; OS_OK = True
    except ImportError:
        OS_OK = False
    raw_dir = Path(args.raw_dir)
    # Try the bare wsi id first (label-stripped) then fall back to the original.
    svs_path = next(
        (raw_dir / f"{cand}{ext}"
         for cand in (_bare_wsi, args.wsi_id)
         for ext in [".svs", ".ndpi", ".tif", ".tiff"]
         if (raw_dir / f"{cand}{ext}").exists()),
        raw_dir / f"{_bare_wsi}.svs",
    )
    thumbnail = None; slide_dims = None; downsample = 1.0
    patch_spacing_px = compute_patch_spacing(str(svs_path), target_patch_um=args.target_patch_um) \
                       if svs_path.exists() else int(args.target_patch_um * (0.5 / 0.25))
    if OS_OK and svs_path.exists():
        slide = openslide.OpenSlide(str(svs_path))
        slide_dims = slide.level_dimensions[0]
        # Prefer the requested vis_level (default 1). If unavailable, fall back
        # to the highest available level and log it.
        if args.vis_level >= slide.level_count:
            lvl = slide.level_count - 1
            print(f"[thumb] requested vis_level={args.vis_level} unavailable "
                  f"(level_count={slide.level_count}); fallback to level={lvl} "
                  f"for {args.wsi_id}")
        else:
            lvl = args.vis_level
        downsample = slide.level_downsamples[lvl]
        tw, th = slide.level_dimensions[lvl]
        thumbnail = np.array(slide.get_thumbnail((tw, th)).convert("RGB"))
        slide.close()
        print(f"[thumb] level={lvl} thumbnail={tw}x{th} downsample={downsample:.2f}")
        print(f"[thumb] chosen vis_level={lvl}  thumb=({tw}x{th})  "
              f"downsample={downsample:.2f}  patch_spacing_l0={patch_spacing_px}  "
              f"(PDF physical width unchanged: 89 mm)")
    else:
        print(f"[warn] svs not found at {svs_path}; no thumbnail")

    cv_arr, half, sw_px, sh_px, pv = _vis_coords(coords_np, slide_dims, downsample, patch_spacing_px)
    cv_x = cv_arr[:, 0]; cv_y = cv_arr[:, 1]

    def fig_size():
        # All spatial maps target 89 mm (single column) wide; height follows
        # the slide aspect ratio but is clamped to <= 4 in for page balance.
        fw = TARGET_W_IN
        ratio = (sh_px / sw_px) if sw_px > 0 else 1.0
        fh = min(fw * ratio, 4.0)
        return fw, fh

    def marker_size():
        fw, _ = fig_size()
        return _ms_helper(pv, fw, sw_px, args.dpi)

    def base_axes():
        fw, fh = fig_size()
        fig, ax = plt.subplots(figsize=(fw, fh))
        if thumbnail is not None:
            # Thumbnail background alpha: spec target 0.95.
            ax.imshow(thumbnail, extent=[0, sw_px, sh_px, 0], aspect="equal",
                      alpha=0.95, interpolation='lanczos')
        ax.set_xlim(0, sw_px); ax.set_ylim(sh_px, 0)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor("white")
        return fig, ax

    def style_colorbar(cb, label_text):
        # Smaller, tighter colorbar for evidence/contribution maps.
        # Spec: frame 0.5 pt black; ticks 6 pt Arial.
        cb.ax.tick_params(labelsize=6, width=0.5, length=2)
        for _t in cb.ax.get_yticklabels():
            _t.set_fontfamily('sans-serif')
        cb.set_label(label_text, fontsize=8, rotation=90, labelpad=6)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor('black')

    DPI = args.dpi

    # 1. Concept label heatmap (3-class) + legend ------------------------------
    fig, ax = base_axes()
    ms = marker_size()
    cmap3 = ListedColormap(GROUP3_COLORS)
    ax.scatter(cv_x + half, cv_y + half, c=group3, cmap=cmap3, vmin=-0.5, vmax=2.5,
               s=ms, alpha=0.85, edgecolors="none", marker="s")
    counts3 = np.array([int((group3 == g).sum()) for g in range(3)])
    pcts3 = counts3 / N * 100
    # Legend intentionally omitted; rendered as a separate file
    # (concept_label_legend.pdf) below.

    # 1 mm scale bar (bottom-right): length in level-0 pixels / downsample.
    # patch_spacing_px corresponds to target_patch_um μm at level 0, so:
    #   px_per_um_l0 = patch_spacing_px / target_patch_um
    #   1 mm = 1000 μm at level 0 -> divide by downsample to get vis-level px.
    try:
        px_per_um_l0 = patch_spacing_px / float(args.target_patch_um)
        bar_len_vis = (1000.0 * px_per_um_l0) / float(downsample if downsample else 1.0)
        bar_h = max(sh_px * 0.006, 2.0)
        margin_x = sw_px * 0.04
        margin_y = sh_px * 0.06
        x0_bar = sw_px - margin_x - bar_len_vis
        y0_bar = sh_px - margin_y - bar_h
        ax.add_patch(Rectangle((x0_bar, y0_bar), bar_len_vis, bar_h,
                               facecolor="white", edgecolor="black",
                               linewidth=0.6, zorder=10))
        ax.text(x0_bar + bar_len_vis / 2, y0_bar + bar_h + sh_px * 0.018,
                "1 mm", ha="center", va="top", fontsize=8, color="black",
                zorder=11)
    except Exception as e:
        print(f"[scalebar] skipped: {e}")

    plt.tight_layout(pad=0)
    # Assert: heatmap PDF has no legend artist before saving.
    assert ax.get_legend() is None, \
        "concept_label_heatmap must not carry a legend artist"
    _save_pdf(fig, out_dir, "concept_label_heatmap"); plt.close(fig)

    # 1b. Standalone legend for concept_label_heatmap --------------------------
    fig_leg, ax_leg = plt.subplots(figsize=(TARGET_W_IN, 12 / MM_PER_INCH))
    ax_leg.axis('off')
    legend_handles = [
        Patch(facecolor=CODELETED,     edgecolor='white',
              label=f"1p/19q codeleted   {pcts3[0]:.1f}%"),
        Patch(facecolor=NON_CODELETED, edgecolor='white',
              label=f"1p/19q non-codeleted   {pcts3[1]:.1f}%"),
        Patch(facecolor=OTHERS,        edgecolor='white',
              label=f"others   {pcts3[2]:.1f}%"),
    ]
    leg = ax_leg.legend(
        handles=legend_handles, loc='center', ncol=3,
        frameon=False, fontsize=10,
        handlelength=1.4, handletextpad=0.5,
        columnspacing=2.5, borderpad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontfamily('sans-serif')
    fig_leg.savefig(out_dir / "concept_label_legend.pdf",
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_leg)
    # Assertions on standalone legend file.
    _legend_path = out_dir / "concept_label_legend.pdf"
    assert _legend_path.exists() and _legend_path.stat().st_size > 0, \
        "concept_label_legend.pdf missing or empty"
    assert sum(1 for h in legend_handles if isinstance(h, Patch)) == 3, \
        "legend must contain exactly 3 Patch handles"
    print(f"[legend] wrote {_legend_path.name} "
          f"({_legend_path.stat().st_size} B, 3 Patch handles)")

    # 2. Concept-count bar (6 horizontal bars, same-class same-colour) --------
    # Horizontal layout (barh). Widened to 130 mm (1.5-column) so the longest
    # labels (chicken-wire vessels, fried-egg cytoplasm, pleomorphic nuclei)
    # fit inside the left margin without clipping. Soft-wrap the two longest
    # labels onto two lines for cleaner reading. Compact 3-entry legend ABOVE.
    WIDE_W_IN = 130.0 / MM_PER_INCH
    bar_figsize = (WIDE_W_IN, 78.0 / MM_PER_INCH)   # ~130 x 78 mm
    fig, ax = plt.subplots(figsize=bar_figsize)
    counts6 = np.array([int((category == c).sum()) for c in range(6)])
    bar_colors = [CODELETED] * N_OLIGO + [NON_CODELETED] + [OTHERS]
    labels = [
        "round nuclei",
        "fried-egg\ncytoplasm",
        "chicken-wire\nvessels",
        "microcalcification",
        "pleomorphic\nnuclei",
        "others",
    ]
    y_pos = np.arange(6)[::-1]                    # 0 at bottom; first cat at top
    # Bar styling: opaque fill, thin black edge per spec (Change 7).
    bars = ax.barh(y_pos, counts6, color=bar_colors,
                   edgecolor='black', linewidth=0.5, height=0.65, alpha=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontfamily='sans-serif', ma='right')
    ax.set_xlabel("Patch count", fontsize=10, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=9)

    max_count = float(counts6.max()) if counts6.max() > 0 else 1.0

    # Count annotation right of each bar.
    for y, c in zip(y_pos, counts6):
        ax.text(c + max_count * 0.012, y, str(int(c)),
                va='center', ha='left', fontsize=9, color='black',
                fontfamily='sans-serif')
        print(f"[bar ] y={y} count={int(c)} -> x={c + max_count*0.012:.2f} "
              f"(right, black)")

    ax.set_xlim(0, max_count * 1.18)

    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color('black')

    handles = [
        Patch(facecolor=CODELETED,     edgecolor='white', label='1p/19q codeleted'),
        Patch(facecolor=NON_CODELETED, edgecolor='white', label='1p/19q non-codeleted'),
        Patch(facecolor=OTHERS,        edgecolor='white', label='others'),
    ]
    leg = ax.legend(handles=handles, loc='upper center',
                    bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
                    fontsize=8, handlelength=1.0, handletextpad=0.35,
                    columnspacing=1.0, borderpad=0.0)

    fig.subplots_adjust(left=0.32, right=0.97, top=0.84, bottom=0.16)

    # Force renderer realisation so window_extent queries are accurate.
    fig.canvas.draw()

    # Width-budget verification: figure must be ~130 x 78 mm.
    _size_mm = fig.get_size_inches() * MM_PER_INCH
    print(f"[bar ] fig size = {_size_mm[0]:.1f} x {_size_mm[1]:.1f} mm "
          f"(target 130 x 78 mm)")

    # Assert no y-tick label is clipped on the left side of the figure.
    _renderer = fig.canvas.get_renderer()
    for tick in ax.get_yticklabels():
        bb = tick.get_window_extent(renderer=_renderer).transformed(
            fig.transFigure.inverted())
        print(f"[bar ] ytick {tick.get_text()!r:<30} fig.x0={bb.x0:.3f}")
        assert bb.x0 > 0.01, \
            f"Label clipped: {tick.get_text()!r} x0={bb.x0:.3f}"

    # Assertions
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert yticklabels == labels, \
        f"y-tick labels must equal full names; got {yticklabels}"
    assert ax.get_legend() is not None, \
        "concept_count_bar must carry a legend artist"
    leg_handles = ax.get_legend().legend_handles
    assert len(leg_handles) == 3, \
        f"concept_count_bar legend must have 3 handles, got {len(leg_handles)}"
    expected_colors = [CODELETED, NON_CODELETED, OTHERS]
    for i, (h, exp) in enumerate(zip(leg_handles, expected_colors)):
        assert isinstance(h, Patch), \
            f"legend handle {i} must be Patch, got {type(h).__name__}"
        actual = mpl.colors.to_hex(h.get_facecolor()).upper()
        assert actual == exp.upper(), \
            f"legend handle {i} color={actual} expected {exp}"
    # Verify no bracket-style figure artists remain.
    n_line2d_artists = sum(1 for a in fig.artists if isinstance(a, Line2D))
    assert n_line2d_artists == 0, \
        f"concept_count_bar: unexpected {n_line2d_artists} Line2D fig.artists"
    n_fig_text = len([t for t in fig.texts])
    assert n_fig_text == 0, \
        f"concept_count_bar: unexpected {n_fig_text} fig.text artists"
    # Bar count text must lie within the x-axis limit (with margin).
    x_lim_max = ax.get_xlim()[1]
    for c in counts6:
        text_x = c + max_count * 0.012
        assert text_x <= x_lim_max, \
            f"count text x={text_x} exceeds x_lim={x_lim_max}"
    print(f"[bar ] legend handles (3): "
          f"{[mpl.colors.to_hex(h.get_facecolor()).upper() for h in leg_handles]}")
    print(f"[bar ] y-tick labels: {yticklabels}")
    print(f"[bar ] no bracket artists  Line2D={n_line2d_artists}  "
          f"fig.texts={n_fig_text}  OK")
    print(f"[bar ] figsize = {bar_figsize[0]*MM_PER_INCH:.1f} x "
          f"{bar_figsize[1]*MM_PER_INCH:.1f} mm  (horizontal barh)")

    _save_pdf(fig, out_dir, "concept_count_bar"); plt.close(fig)

    # 2b. Concept-contribution bar — share of total diagnostic evidence ------
    # Per-patch evidence: ev[i, k] = attn_weights[i, k] × concept_scores[i, k]
    #                                  × concept_scale[k]
    # Per-concept assigned contribution (Σ over patches with category == k):
    #     contrib_k = Σ_{i: category==k} ev[i, k]
    # Bar height = contrib_k / Σ_j contrib_j × 100, i.e. each concept's share
    # of the slide's total diagnostic evidence. The 5 bars sum to 100%.
    # "others" patches are saliency-gated out of the slide profile and have no
    # concept-bound contribution — they are omitted.
    contrib5 = np.array([float(ev_np[category == k, k].sum()) for k in range(N_DIAG)])
    total_ev = float(contrib5.sum())
    if total_ev > 0:
        contrib_pct = contrib5 / total_ev * 100
    else:
        contrib_pct = np.zeros_like(contrib5)
    print(f"[bar2] per-concept assigned contribution (total Σev = {total_ev:.4f}):")
    for k, name in enumerate(CONCEPT_NAMES):
        n_assigned = int((category == k).sum())
        print(f"        {name:<22} n={n_assigned:>4}  Σev={contrib5[k]:.4f}  "
              f"share={contrib_pct[k]:5.1f}%")

    fig2, ax2 = plt.subplots(figsize=bar_figsize)
    bar_colors5 = [CODELETED] * N_OLIGO + [NON_CODELETED]
    labels5 = labels[:N_OLIGO + 1]                 # drop "others" from count_bar labels
    y5 = np.arange(N_DIAG)[::-1]
    ax2.barh(y5, contrib_pct, color=bar_colors5,
             edgecolor='black', linewidth=0.5, height=0.65, alpha=1.0)
    ax2.set_yticks(y5)
    ax2.set_yticklabels(labels5, fontsize=9, fontfamily='sans-serif', ma='right')
    ax2.set_xlabel("Share of diagnostic contribution (%)",
                   fontsize=10, fontfamily='sans-serif')
    ax2.tick_params(axis='x', labelsize=9)

    max_c = float(contrib_pct.max()) if contrib_pct.max() > 0 else 1.0
    for y_pos2, p in zip(y5, contrib_pct):
        ax2.text(p + max_c * 0.012, y_pos2, f"{p:.1f}%",
                 va='center', ha='left', fontsize=9, color='black',
                 fontfamily='sans-serif')
    ax2.set_xlim(0, max_c * 1.18)
    for spine in ax2.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color('black')

    # 2-entry legend (no "others" — others has no contribution).
    handles_contrib = [
        Patch(facecolor=CODELETED,     edgecolor='white', label='1p/19q codeleted'),
        Patch(facecolor=NON_CODELETED, edgecolor='white', label='1p/19q non-codeleted'),
    ]
    ax2.legend(handles=handles_contrib, loc='upper center',
               bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False,
               fontsize=8, handlelength=1.0, handletextpad=0.35,
               columnspacing=1.0, borderpad=0.0)
    fig2.subplots_adjust(left=0.32, right=0.97, top=0.84, bottom=0.16)

    _save_pdf(fig2, out_dir, "concept_contribution_bar"); plt.close(fig2)
    print(f"[bar2] saved concept_contribution_bar.pdf   "
          f"figsize ~ {bar_figsize[0]*MM_PER_INCH:.1f} x {bar_figsize[1]*MM_PER_INCH:.1f} mm")

    # 3-6. Per-concept maps (MorphoPath cmap for oligo, warm-teal ASTRO for astro) ---
    # TP-mode: skip the astro concept map (k=4); keep only the 4 oligo concepts.
    # OVERLAP MODE: drop the argmax (`category == k`) restriction. For each
    # concept, colour EVERY diagnostic patch by its attn for that concept;
    # a patch can therefore appear bright in multiple concept maps if it has
    # multi-concept attention. Non-diagnostic patches (is_diag == False) stay
    # as the light-grey background.
    tp_mode = (args.label.upper() == "TP")
    tn_mode = (args.label.upper() == "TN")
    # OVERLAP_SAL: heatmap intensity = attn_specific (mean-subtracted attn_np).
    # Diverging colormap (RdBu_r) with SYMMETRIC vmin/vmax around 0 so:
    #   red  = patch attends to this concept MORE than the 4-oligo avg
    #   blue = patch attends to this concept LESS than avg
    #   white = patch is concept-neutral (attn equal across concepts)
    # vabs = max(|percentile(1)|, |percentile(99)|) over all 4 oligo concepts
    # × diagnostic patches — global symmetric scale, cross-concept comparable.
    if display_mask.any():
        _z_vals = morph_zcross[display_mask, :].ravel()
        _vabs_oligo = max(abs(np.percentile(_z_vals, 1)),
                          abs(np.percentile(_z_vals, 99)), 1e-12)
    else:
        _vabs_oligo = 1e-12
    print(f"[overlap_morphzcross-norm] global oligo |vabs| (morph_zcross) = {_vabs_oligo:.4f} "
          f"(diverging cmap symmetric around 0; values in [-vabs, +vabs])")
    for k, name in enumerate(CONCEPT_NAMES):
        if tp_mode and k == N_OLIGO:
            continue   # skip astro_nuclear in TP mode
        if tn_mode and k != N_OLIGO:
            continue   # TN mode: render ONLY astro_nuclear (k=4)
        fig, ax = base_axes(); ms = marker_size()
        not_disp = ~display_mask
        if not_disp.any():
            ax.scatter(cv_x[not_disp] + half, cv_y[not_disp] + half,
                       c="lightgray", s=ms * 0.4, alpha=0.10,
                       edgecolors="none", marker="s")
        if display_mask.any():
            if k < N_OLIGO:
                vals = morph_zcross[display_mask, k]
                sc = ax.scatter(cv_x[display_mask] + half, cv_y[display_mask] + half,
                                c=vals, cmap="RdBu_r",
                                vmin=-_vabs_oligo, vmax=+_vabs_oligo,
                                s=ms, alpha=0.85, edgecolors="none", marker="s")
                cb = plt.colorbar(sc, ax=ax, shrink=0.25, pad=0.015, aspect=25)
                style_colorbar(cb, "morph z (across slide)")
            elif tn_mode and k == N_OLIGO:
                # TN: astro morphzcross (single-column z across patches).
                _av = morph_zcross_astro[display_mask]
                vabs_a = max(abs(np.percentile(_av, 1)),
                             abs(np.percentile(_av, 99)), 1e-12)
                sc = ax.scatter(cv_x[display_mask] + half, cv_y[display_mask] + half,
                                c=_av, cmap="RdBu_r",
                                vmin=-vabs_a, vmax=+vabs_a,
                                s=ms, alpha=0.85, edgecolors="none", marker="s")
                cb = plt.colorbar(sc, ax=ax, shrink=0.25, pad=0.015, aspect=25)
                style_colorbar(cb, "morph z (across slide)")
            else:
                vals = attn_np[is_diag, k]
                if vals.max() > 0:
                    vmax_use = max(np.percentile(vals, 99), 1e-12)
                    sc = ax.scatter(cv_x[is_diag] + half, cv_y[is_diag] + half,
                                    c=vals, cmap=ASTRO_CMAP,
                                    vmin=0, vmax=vmax_use,
                                    s=ms, alpha=0.85, edgecolors="none", marker="s")
                    cb = plt.colorbar(sc, ax=ax, shrink=0.25, pad=0.015, aspect=25)
                    style_colorbar(cb, "Attn")
        plt.tight_layout(pad=0)
        if tp_mode and k < N_OLIGO:
            pdf_name = f"concept_{k}_{name}"
        elif tn_mode and k == N_OLIGO:
            pdf_name = f"concept_{k}_{name}"   # concept_4_astro_nuclear.pdf
        else:
            pdf_name = f"map_{name}"
        _save_pdf(fig, out_dir, pdf_name); plt.close(fig)

    # 7. Others map (1 - saliency) --------------------------------------------
    # Palette choice: use the module-level OTHERS_CMAP (white -> #91D1C2),
    # which lands exactly on the OTHERS hue and blends with the coral/cerulean/mint
    # 3-class palette better than matplotlib's 'Greens'. Print the choice.
    # TP-mode: skip the others map (oligo-class concept context not needed).
    if tp_mode:
        print("[cmap] TP-mode: skipping map_others.pdf")
    else:
        print("[cmap] others map: using palette-coherent OTHERS_CMAP "
              "(white -> #91D1C2) instead of matplotlib 'Greens' for hue match")
        fig, ax = base_axes(); ms = marker_size()
        is_o = (category == 5)
        if (~is_o).any():
            ax.scatter(cv_x[~is_o] + half, cv_y[~is_o] + half, c="lightgray",
                       s=ms * 0.4, alpha=0.10, edgecolors="none", marker="s")
        if is_o.any():
            vals = 1.0 - saliency[is_o]
            sc = ax.scatter(cv_x[is_o] + half, cv_y[is_o] + half, c=vals,
                            cmap=OTHERS_CMAP, vmin=vals.min(), vmax=1.0,
                            s=ms, alpha=0.85, edgecolors="none", marker="s")
            cb = plt.colorbar(sc, ax=ax, shrink=0.25, pad=0.015, aspect=25)
            style_colorbar(cb, "Non-diagnostic confidence")
        plt.tight_layout(pad=0)
        _save_pdf(fig, out_dir, "map_others"); plt.close(fig)

    # 8. Model contribution (independent of per-concept) -----------------------
    pred = 1 if prob > 0.5 else 0
    if pred == 1:
        contrib = ev_np[:, :N_OLIGO].sum(axis=1).copy(); cmap_use = MORPHOPATH_CMAP
    else:
        contrib = ev_np[:, N_OLIGO:N_DIAG].sum(axis=1).copy(); cmap_use = ASTRO_CMAP
    # OVERLAP v11.7: apply the same sal upper cap (default 0.97) to
    # model_contribution so this figure is consistent with concept_label_heatmap
    # and the 4 concept heatmaps. Cap-dropped patches get contrib=0 (white).
    if sal_upper < float('inf'):
        _cap_mask = saliency >= sal_upper
        contrib[_cap_mask] = 0.0
        print(f"[model_contribution] zeroed {_cap_mask.sum()} cap-dropped (sal>=sal_upper) patches")
    fig, ax = base_axes(); ms = marker_size()
    if contrib.max() > 0:
        vmax = max(np.percentile(contrib, 99), 1e-12)
        sc = ax.scatter(cv_x + half, cv_y + half, c=contrib, cmap=cmap_use,
                        vmin=0, vmax=vmax, s=ms, alpha=0.85,
                        edgecolors="none", marker="s")
        cb = plt.colorbar(sc, ax=ax, shrink=0.25, pad=0.015, aspect=25)
        style_colorbar(cb, "Concept evidence")
    plt.tight_layout(pad=0)
    _save_pdf(fig, out_dir, "model_contribution"); plt.close(fig)

    # --- Top-N patches per category (PNG @ slide pyramid level 1) -------------
    # Coords stored in coords.csv refer to slide level 0 (unchanged); the saved
    # crop is read at level 1 (sharper than level 2) with fallback to level 0
    # if the slide pyramid has fewer than 2 levels.
    PATCH_LEVEL = 1

    # Precompute global oligo rank maps per concept k in {0,1,2,3}.
    # Used by TP mode to annotate coords.csv with `oligo_rank_in_slide`.
    rank_maps = {}
    # OVERLAP_SAL v11: rank by attn_np[:, k] over ALL diagnostic patches
    # (matches new save_top ranking logic, before diversification).
    for k in range(N_OLIGO):
        diag_idxs = np.where(is_pick)[0]
        attn_k_diag = attn_np[is_pick, k]
        order_k = np.argsort(attn_k_diag)[::-1]
        rank_maps[k] = {int(diag_idxs[o]): r + 1 for r, o in enumerate(order_k)}

    is_tp = (args.label.upper() == "TP")

    def save_top(cat_idx, name, scores, descending=True, prepicked=None, target_dir=None):
        _tdir = target_dir if target_dir is not None else top_dir
        cdir = _tdir / name; cdir.mkdir(parents=True, exist_ok=True)
        # Clean stale PDF/PNG crops from previous runs so the per-category
        # output reflects only the current invocation (and the PNG-only
        # mechanical assertion is stable across re-runs).
        for old in list(cdir.glob("*.pdf")) + list(cdir.glob("*.png")):
            try: old.unlink()
            except Exception: pass
        # OVERLAP_SAL v11: caller passes `prepicked` (greedy-diversified
        # patch indices) for oligo concepts in TP mode. For other paths
        # (astro, others, non-TP) we fall back to the original sort logic.
        if prepicked is not None:
            picked = np.asarray(prepicked, dtype=int)
            idxs = np.where(is_pick)[0]   # for logging only
        else:
            if cat_idx == 5:
                idxs = np.arange(len(scores))
            else:
                idxs = np.where(is_pick)[0]
            if len(idxs) == 0:
                print(f"  [top] {name}: no patches"); return
            order = np.argsort(scores[idxs])[::-1] if descending else np.argsort(scores[idxs])
            picked = idxs[order[:args.top_n]]
        rows = []
        slide = openslide.OpenSlide(str(svs_path)) if (OS_OK and svs_path.exists()) else None
        # Resolve patch read level + size once per category.
        if slide is not None:
            lvl = PATCH_LEVEL if slide.level_count >= 2 else 0
            if lvl != PATCH_LEVEL:
                print(f"  [top] fallback to level={lvl} for {args.wsi_id} "
                      f"(level_count={slide.level_count} < 2)")
            ds = slide.level_downsamples[lvl]
            lvl_size = max(1, int(round(patch_spacing_px / ds)))
        else:
            lvl, lvl_size = None, None
        # In TP mode, attach `oligo_rank_in_slide` only when this category is
        # one of the 4 oligo concepts.
        attach_oligo_rank = is_tp and (cat_idx in rank_maps)
        for rank, i in enumerate(picked, 1):
            x0 = int(coords_np[i, 0]); y0 = int(coords_np[i, 1])
            ev_val = float(ev_np[i, cat_idx]) if cat_idx < 5 else float("nan")
            attn_val = float(attn_np[i, cat_idx]) if cat_idx < 5 else float("nan")
            share_val = float(concept_share[i, cat_idx]) if cat_idx < 5 else float("nan")
            ma_val    = float(morph_attn[i, cat_idx]) if cat_idx < 5 else float("nan")
            spec_val  = float(attn_specific[i, cat_idx]) if cat_idx < N_OLIGO else float("nan")
            z_val     = float(attn_z[i, cat_idx]) if cat_idx < N_OLIGO else float("nan")
            morph_z_val = float(morph_zcross[i, cat_idx]) if cat_idx < N_OLIGO else float("nan")
            in_disp_val = bool(display_mask[i]) if cat_idx < N_OLIGO else False
            # OVERLAP MODE: flag whether this patch's argmax winner is this
            # concept (True) or a different concept (False). False = overlap
            # candidate — patch has high share for THIS concept but its argmax
            # winner is some other concept.
            argmax_winner = bool(int(category[i]) == int(cat_idx)) if cat_idx < 5 else False
            row = {"rank": rank, "patch_idx": int(i),
                   "coord_x_pixel": x0, "coord_y_pixel": y0,
                   "saliency": float(saliency[i]),
                   "morph_zcross": morph_z_val,
                   "in_display_mask": in_disp_val,
                   "attn_z": z_val,
                   "attn_specific": spec_val,
                   "morph_attn": ma_val,
                   "concept_share": share_val,
                   "attn_value": attn_val,
                   "ev_value": ev_val,
                   "score_used": float(scores[i]),
                   "is_argmax_winner": argmax_winner,
                   "argmax_concept": int(category[i])}
            if attach_oligo_rank:
                row["oligo_rank_in_slide"] = rank_maps[cat_idx].get(int(i), "N/A")
            rows.append(row)
            if slide is not None:
                try:
                    img = slide.read_region((x0, y0), lvl,
                                            (lvl_size, lvl_size)).convert("RGB")
                    if _REINHARD is not None:
                        img = _REINHARD(img)
                    img.save(cdir / f"rank{rank:02d}.png", "PNG",
                             optimize=True, compress_level=6)
                except Exception as e:
                    print(f"    rank{rank} fail: {e}")
        fieldnames = ["rank", "patch_idx",
                      "coord_x_pixel", "coord_y_pixel",
                      "saliency", "morph_zcross", "in_display_mask", "attn_z", "attn_specific", "morph_attn", "concept_share",
                      "attn_value", "ev_value", "score_used",
                      "is_argmax_winner", "argmax_concept"]
        if attach_oligo_rank:
            # Insert the new column directly after patch_idx for readability.
            fieldnames = ["rank", "patch_idx", "oligo_rank_in_slide",
                          "coord_x_pixel", "coord_y_pixel",
                          "saliency", "morph_zcross", "in_display_mask", "attn_z", "attn_specific", "morph_attn", "concept_share",
                          "attn_value", "ev_value", "score_used",
                          "is_argmax_winner", "argmax_concept"]
        with open(cdir / "coords.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        # Mechanical assertion: only PNG crops written for this category.
        n_pdf = len(list(cdir.glob("*.pdf")))
        n_png = len(list(cdir.glob("*.png")))
        assert n_pdf == 0, f"{cdir.name}: unexpected {n_pdf} PDF crops"
        assert n_png == len(picked), \
            f"{cdir.name}: expected {len(picked)} PNG crops, found {n_png}"
        print(f"  [top] {name}: saved {n_png}/{len(idxs)} PNG -> {cdir.relative_to(out_dir)}")
        return picked

    # OVERLAP_SAL v11: GREEDY DIVERSIFICATION (plan D).
    # Each concept ranks its candidates by attn_np[:, k] descending. We
    # process concepts round-robin: in each round each concept picks its
    # next-best candidate that hasn't yet been claimed by MAX_OVERLAP=2
    # concepts. This:
    #   - keeps attn_np as the truth signal (no specificity bias)
    #   - allows a patch to appear in up to 2 concept top-N (real multi-
    #     concept patches preserved, but not in all 4 simultaneously)
    #   - avoids forcing 4 top-N to be near-identical (~50% unique in
    #     pure attn_np) — diversification pushes uniqueness up.
    MAX_OVERLAP = 3

    if is_tp:
        # Build per-concept sorted candidate lists (diagnostic patches only,
        # sorted by --rank_by metric descending).
        rank_metric = ev_np if args.rank_by == "ev" else attn_np
        print(f"[rank_by] {args.rank_by} ({'attn × score × scale' if args.rank_by == 'ev' else 'morph_attn × saliency / norm'})")
        diag_idxs = np.where(is_pick)[0]
        cand_per_concept = []
        for k in range(N_OLIGO):
            m_k = rank_metric[diag_idxs, k]
            sorted_diag = diag_idxs[np.argsort(m_k)[::-1]]
            cand_per_concept.append(sorted_diag.tolist())
        pointer = [0] * N_OLIGO
        picks_per_concept = {k: [] for k in range(N_OLIGO)}
        patch_used_count = {}   # patch_idx -> count of how many concepts picked it

        def _pick_next_for(k):
            """Advance concept k's pointer to next legal candidate; return
            patch_idx or None if exhausted."""
            while pointer[k] < len(cand_per_concept[k]):
                p = int(cand_per_concept[k][pointer[k]])
                pointer[k] += 1
                if patch_used_count.get(p, 0) < MAX_OVERLAP:
                    return p
            return None

        # Round-robin until each concept has top_n picks or candidates exhausted.
        any_added = True
        while any_added and any(len(picks_per_concept[k]) < args.top_n for k in range(N_OLIGO)):
            any_added = False
            for k in range(N_OLIGO):
                if len(picks_per_concept[k]) >= args.top_n:
                    continue
                nxt = _pick_next_for(k)
                if nxt is None:
                    continue
                picks_per_concept[k].append(nxt)
                patch_used_count[nxt] = patch_used_count.get(nxt, 0) + 1
                any_added = True

        # Diversification summary print.
        n_shared = sum(1 for c in patch_used_count.values() if c > 1)
        total_picked = sum(len(v) for v in picks_per_concept.values())
        print(f"[diversify] MAX_OVERLAP={MAX_OVERLAP}; total picks="
              f"{total_picked}, unique patches={len(patch_used_count)}, "
              f"{n_shared} shared (in 2 concepts)")

        # Now write outputs using a customised save_top that takes a
        # pre-computed picked list (bypasses the internal sort).
        for k in range(N_OLIGO):
            save_top(k, CONCEPT_NAMES[k], attn_np[:, k], descending=True,
                     prepicked=picks_per_concept[k])

        # ---- v11.8: parallel top-N using PURE morph_attn (saliency-free) ----
        # Output dir: top20_morph/<concept>/. Same greedy MAX_OVERLAP=3.
        # NOTE: pool is still is_pick (so sal upper cap still applies to the
        # candidate set), but the ranking key inside each concept is morph_attn
        # rather than attn_np = morph_attn × saliency. Top-N here is the
        # "would-be picks if saliency didn't influence ranking".
        if not getattr(args, "skip_morph_top", False):
            morph_top_dir = out_dir / "top20_morph"
            morph_top_dir.mkdir(exist_ok=True)
            print("[morph-top] greedy round-robin using pure morph_attn ranking")
            _cand2 = []
            for k in range(N_OLIGO):
                m_k = morph_attn[diag_idxs, k]
                srt = diag_idxs[np.argsort(m_k)[::-1]]
                _cand2.append(srt.tolist())
            _ptr = [0] * N_OLIGO
            _picks = {k: [] for k in range(N_OLIGO)}
            _used = {}
            def _pick2(k):
                while _ptr[k] < len(_cand2[k]):
                    p = int(_cand2[k][_ptr[k]])
                    _ptr[k] += 1
                    if _used.get(p, 0) < MAX_OVERLAP:
                        return p
                return None
            any_added2 = True
            while any_added2 and any(len(_picks[k]) < args.top_n for k in range(N_OLIGO)):
                any_added2 = False
                for k in range(N_OLIGO):
                    if len(_picks[k]) >= args.top_n: continue
                    nxt = _pick2(k)
                    if nxt is None: continue
                    _picks[k].append(nxt)
                    _used[nxt] = _used.get(nxt, 0) + 1
                    any_added2 = True
            n_shared2 = sum(1 for c in _used.values() if c > 1)
            total2 = sum(len(v) for v in _picks.values())
            print(f"[morph-top] MAX_OVERLAP={MAX_OVERLAP}; total picks={total2}, "
                  f"unique={len(_used)}, {n_shared2} shared (>=2 concepts)")
            for k in range(N_OLIGO):
                save_top(k, CONCEPT_NAMES[k], morph_attn[:, k], descending=True,
                         prepicked=_picks[k], target_dir=morph_top_dir)
        else:
            print("[morph-top] skipped (--skip_morph_top)")
    elif tn_mode:
        # TN: single-concept astro top-N (no greedy; only k=4).
        ASTRO_K = N_OLIGO
        pool_idx = np.where(is_pick)[0]
        if len(pool_idx) == 0:
            print("[tn-top] empty is_pick pool")
        else:
            astro_attn = attn_np[pool_idx, ASTRO_K]
            order = np.argsort(astro_attn)[::-1]
            picked = pool_idx[order[:args.top_n]]
            print(f"[tn-top] astro top-{args.top_n}: {len(picked)} from pool {len(pool_idx)}")
            save_top(ASTRO_K, CONCEPT_NAMES[ASTRO_K], attn_np[:, ASTRO_K],
                     descending=True, prepicked=picked)
    else:
        for k, n in enumerate(CONCEPT_NAMES):
            scores = attn_np[:, k] if k < N_DIAG else saliency
            save_top(k, n, scores, descending=True)
        save_top(5, "others", saliency, descending=False)

    # --- Verification figure (TP or TN manual_picks) -------------------------
    if (is_tp or tn_mode) and args.manual_picks.strip():
        try:
            # Build (N, N_DIAG=5) intensity matrix combining oligo morph_zcross
            # (cols 0-3) and astro morph_zcross_astro (col 4) so _render_tp_
            # verification can handle both TP and TN concepts.
            _mz_full = np.zeros((morph_zcross.shape[0], N_DIAG))
            _mz_full[:, :N_OLIGO] = morph_zcross
            _mz_full[:, N_OLIGO] = morph_zcross_astro
            _render_tp_verification(
                args, out_dir, top_dir, thumbnail, cv_x, cv_y, half,
                sw_px, sh_px, ev_np, category, marker_size(),
                patch_spacing_px, downsample,
                intensity_np=_mz_full,
                display_mask=display_mask,
                vabs_symmetric=_vabs_oligo,
                diverging=True,
            )
        except Exception as e:
            print(f"[tp-verify] failed: {e}")
            traceback.print_exc()

    # --- Smoke-test summary --------------------------------------------------
    print("\n[palette] CODELETED={}  NON_CODELETED={}  OTHERS={}".format(
        CODELETED, NON_CODELETED, OTHERS))
    sp_w, sp_h = fig_size()
    print("[figsize-mm] spatial-maps      = {:.1f} x {:.1f} mm  (target W = 89 mm)"
          .format(sp_w * MM_PER_INCH, sp_h * MM_PER_INCH))
    print("[figsize-mm] concept_count_bar = {:.1f} x {:.1f} mm  (horizontal, 1.5-col)".format(
        130.0, 78.0))
    print(f"\n[done] outputs in {out_dir}")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            print(f"  {p.relative_to(out_dir)}   {sz:>8d} B")
        else:
            entries = sorted(p.iterdir())
            n_pdf = sum(1 for q in entries if q.suffix == ".pdf")
            n_jpg = sum(1 for q in entries if q.suffix == ".jpg")
            n_png = sum(1 for q in entries if q.suffix == ".png")
            print(f"  {p.relative_to(out_dir)}/  ({len(entries)} entries)")
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    sub_entries = list(sub.iterdir())
                    n_pdf_sub = sum(1 for q in sub_entries if q.suffix == ".pdf")
                    n_jpg_sub = sum(1 for q in sub_entries if q.suffix == ".jpg")
                    n_png_sub = sum(1 for q in sub_entries if q.suffix == ".png")
                    print(f"    {sub.name}/  pdf={n_pdf_sub}  jpg={n_jpg_sub}  png={n_png_sub}")

    return out_dir


def _read_batch_list(xlsx_path):
    """Return [(wsi_id, label), ...] for rows in xlsx where column 1 == 1.

    The xlsx layout is two columns: folder name (e.g. ``128047_TP``) in the
    first column and a marker (``1`` for selected) in the second column. The
    spec phrases this as "column 1 == 1" referring to the marker column.
    Folder names are split with ``rsplit('_', 1)`` so multi-token IDs such as
    ``189575_5_TP`` parse correctly to wsi_id=189575_5, label=TP.
    """
    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb.active
    out = []
    for row in ws.iter_rows(values_only=True):
        if len(row) < 2 or row[0] is None:
            continue
        marker = row[1]
        if marker != 1:
            continue
        folder = str(row[0]).strip()
        if not folder or "_" not in folder:
            continue
        wsi_id, label = folder.rsplit("_", 1)
        out.append((wsi_id, label))
    return out


def main():
    args = parse_args()
    if args.batch:
        xlsx_path = Path(args.xlsx)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"--batch xlsx not found: {xlsx_path}")
        batch = _read_batch_list(xlsx_path)
        n_total = len(batch)
        print(f"[batch] driver: xlsx={xlsx_path}  selected WSIs={n_total}")
        n_ok = 0
        failures = []
        for idx, (wsi_id, label) in enumerate(batch, 1):
            args.wsi_id = wsi_id
            args.label = label
            if args.out_root:
                args.out_dir = str(Path(args.out_root) / wsi_id)
                out_dir = Path(args.out_dir)
            elif args.out_dir:
                out_dir = Path(args.out_dir)
            elif label.upper() == "TP":
                out_dir = REPO / "visual" / "tp_verification" / wsi_id
            else:
                out_dir = REPO / "visual" / "vis_filtered_nograde" / wsi_id
            print(f"\n[batch] {idx}/{n_total}  wsi={wsi_id}  label={label}  "
                  f"out={out_dir}")
            try:
                run_one(args)
                n_ok += 1
            except Exception as exc:
                failures.append((wsi_id, type(exc).__name__))
                print(f"[batch] FAIL wsi={wsi_id}: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        n_fail = len(failures)
        print(f"\n[batch] complete: {n_ok}/{n_total} OK, {n_fail} failures")
        if failures:
            print("[batch] failures:")
            for wsi_id, et in failures:
                print(f"        ({wsi_id!r}, {et!r})")
    else:
        if not args.wsi_id:
            raise SystemExit("error: wsi_id required when --batch not used")
        run_one(args)


if __name__ == "__main__":
    main()
