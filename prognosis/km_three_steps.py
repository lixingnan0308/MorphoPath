"""Three-step KM analysis on the canonical cohort tables.

PART A (POS series):
    Predicted-positive subset (risk_high_typ == 1) drawn from cohort_table_all.csv,
    grouped by TRUE label (label_1p19q) -> TP (1) vs FP (0).
    Figures: km_pos_step{1,2,3}_*.

PART B (ALL series):
    Full cohort (cohort_table_all.csv), grouped by model prediction
    (risk_high_typ) or by true label (step 4).
    Figures: km_all_step{1,2,3,4}_*.

All KM curves treat each slide as an independent observation (no patient
cluster correction). `-2` / `-3` suffixed slides are independent patients
(handled upstream in build_cohort).

Figures styled for Nature / Nature Medicine submission:
  - Arial, font.size=7 (axes 7, tick 6, panel labels bold 8)
  - 89 mm single-col width
  - linewidths: curves 1.1 pt, spines/ticks 0.5 pt
  - pdf.fonttype=42 (editable TrueType), svg.fonttype=none
  - legend frame OFF (frameon=False), transparent text block
  - no median-OS annotations (removed)
  - no dashed reference lines
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent
DATA_ALL = ROOT / "data/cohort_table_all.csv"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
FIG_CUT085 = FIG / "predpos_cutoff_085"; FIG_CUT085.mkdir(exist_ok=True)
FIG_CUT095 = FIG / "predpos_cutoff_095"; FIG_CUT095.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

# ---- Nature-grade rcParams ------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "lines.linewidth": 1.1,
    "lines.solid_capstyle": "round",
})

MM = 1 / 25.4  # mm -> inches helper


def _fmt_p(p):
    if pd.isna(p):
        return "P = n/a"
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


# ---------------------------------------------------------------------------
# Corner-density driven legend placement (mOS placement removed)
# ---------------------------------------------------------------------------

CORNER_BOXES = {
    "upper right": (0.58, 1.00, 0.66, 1.00),
    "upper left":  (0.00, 0.42, 0.66, 1.00),
    "lower right": (0.58, 1.00, 0.00, 0.34),
    "lower left":  (0.00, 0.42, 0.00, 0.34),
}
CORNER_ANCHORS = {
    "upper right": (0.98, 0.98),
    "upper left":  (0.02, 0.98),
    "lower right": (0.98, 0.04),
    "lower left":  (0.02, 0.04),
}


def _data_to_axes_y(y_data, Y0=-0.02, Y1=1.08):
    return (y_data - Y0) / (Y1 - Y0)


def _curve_y_at_axes(t_arr, s_arr, x_axes, t_max):
    if len(t_arr) == 0:
        return _data_to_axes_y(1.0)
    x_data = x_axes * t_max
    idx = np.searchsorted(t_arr, x_data, side="right") - 1
    idx = max(0, min(idx, len(s_arr) - 1))
    return _data_to_axes_y(float(s_arr[idx]))


def _box_curve_density(surv_curves, t_max, box, n_samples=24):
    x0, x1, y0, y1 = box
    xs = np.linspace(x0, x1, n_samples)
    total = 0
    inside = 0
    for _v, (t_a, s_a) in surv_curves.items():
        for x_ax in xs:
            y_ax = _curve_y_at_axes(t_a, s_a, x_ax, t_max)
            total += 1
            if y0 <= y_ax <= y1:
                inside += 1
    return (inside / total) if total else 1.0


def _pick_legend_corner(surv_curves, t_max, density_thresh=0.18):
    scores = {name: _box_curve_density(surv_curves, t_max, box)
              for name, box in CORNER_BOXES.items()}
    best = min(scores, key=scores.get)
    if scores[best] > density_thresh and min(scores.values()) > density_thresh:
        return ("outside_right", (1.02, 1.0), True)
    return (best, CORNER_ANCHORS[best], False)


def km_panel(ax_km, ax_tab, sub_df, group_col, group_meta, title,
             t_max=80, t_step=20, t_step_risk=None,
             legend_group_title="Model prediction", title_fontsize=7,
             legend_order=None):
    """Render KM panel + risk table. group_meta = {value: (label, color)}.

    legend_order: optional list of label strings indicating preferred top-to-bottom
                  order in the legend.
    """
    groups = [(val, lab, col, sub_df[sub_df[group_col] == val])
              for val, (lab, col) in group_meta.items()]

    g0 = groups[0][3]; g1 = groups[1][3]
    if len(g0) and len(g1):
        lr = logrank_test(g0["OS_months"], g1["OS_months"],
                          event_observed_A=g0["event"], event_observed_B=g1["event"])
        pval = lr.p_value
    else:
        pval = float("nan")

    surv_curves = {}
    for val, label, color, sub in groups:
        if not len(sub):
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["OS_months"], event_observed=sub["event"], label=label)
        kmf.plot_survival_function(
            ax=ax_km, ci_show=False, color=color, lw=1.1,
            show_censors=True,
            censor_styles={"marker": "|", "ms": 3.5, "mew": 0.6})
        sf = kmf.survival_function_
        t_arr = np.asarray(sf.index.values, dtype=float)
        s_arr = np.asarray(sf.iloc[:, 0].values, dtype=float)
        surv_curves[val] = (t_arr, s_arr)

    ax_km.set_xlim(0, t_max); ax_km.set_ylim(-0.02, 1.08)
    ax_km.set_xticks(range(0, t_max + 1, t_step))
    ax_km.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_km.set_yticklabels(["0", "25", "50", "75", "100"])
    ax_km.set_ylabel("Survival probability (%)", labelpad=2)
    ax_km.set_xlabel("Survival times (months)", labelpad=2)
    ax_km.tick_params(pad=2)

    leg_loc, leg_anchor, leg_outside = _pick_legend_corner(surv_curves, t_max)

    handles, labels = ax_km.get_legend_handles_labels()
    if legend_order is not None:
        pair = list(zip(handles, labels))
        pair.sort(key=lambda hl: (legend_order.index(hl[1])
                                  if hl[1] in legend_order else 99))
        if pair:
            handles, labels = zip(*pair)

    from matplotlib.lines import Line2D
    p_handle = Line2D([0], [0], color="none", marker="",
                      linestyle="", label=f"Log-rank {_fmt_p(pval)}")
    handles_full = list(handles) + [p_handle]
    labels_full = list(labels) + [f"Log-rank {_fmt_p(pval)}"]

    if leg_outside:
        leg = ax_km.legend(handles_full, labels_full,
                           title=legend_group_title,
                           loc="upper left", frameon=False,
                           handlelength=1.2, handletextpad=0.5,
                           borderpad=0.2, columnspacing=0.5,
                           borderaxespad=0.0, labelspacing=0.3,
                           bbox_to_anchor=leg_anchor)
    else:
        leg = ax_km.legend(handles_full, labels_full,
                           title=legend_group_title,
                           loc=leg_loc, frameon=False,
                           handlelength=1.2, handletextpad=0.5,
                           borderpad=0.2, columnspacing=0.5,
                           borderaxespad=0.4, labelspacing=0.3,
                           bbox_to_anchor=leg_anchor)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(6)
        leg.get_texts()[-1].set_color("#222222")
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(6)
            leg.get_title().set_ha("left")

    ax_km.set_title(title, fontsize=title_fontsize, weight="normal",
                    loc="center", pad=6)

    # ---- number-at-risk table -------------------------------------------
    risk_step = t_step_risk if t_step_risk is not None else t_step
    times = list(range(0, t_max + 1, risk_step))
    ax_tab.set_xlim(0, t_max); ax_tab.set_ylim(0, 3)
    ax_tab.set_xticks(times)
    ax_tab.set_xticklabels([str(t) for t in times])
    ax_tab.tick_params(axis="x", pad=1, length=2)
    ax_tab.set_yticks([])
    for sp in ax_tab.spines.values():
        sp.set_visible(False)
    ax_tab.set_xlabel("Survival times (months)", labelpad=2)

    row_y_data = [1.85, 0.85]
    ax_tab.annotate("Number at risk", xy=(0, row_y_data[0]),
                    xycoords=("axes fraction", "data"),
                    xytext=(-10, 16), textcoords="offset points",
                    fontsize=6, weight="bold", ha="right", va="bottom")
    for (val, label, color, sub), row_y in zip(groups, row_y_data):
        short = label
        parts = short.split(" ", 1)
        if len(parts) == 2 and len(parts[1]) <= 10:
            two_line = f"{parts[0]}\n{parts[1]}"
        else:
            two_line = short
        ax_tab.annotate(two_line, xy=(0, row_y),
                        xycoords=("axes fraction", "data"),
                        xytext=(-10, 0), textcoords="offset points",
                        fontsize=6, color=color, ha="right", va="center",
                        linespacing=1.0)
        for t in times:
            at_risk = int((sub["OS_months"] >= t).sum())
            ax_tab.text(t, row_y, str(at_risk),
                        fontsize=5.5, color=color,
                        ha="center", va="center")

    return {"n_total": len(sub_df), "events_total": int(sub_df["event"].sum()),
            "n_group0": len(g0), "events_group0": int(g0["event"].sum()),
            "n_group1": len(g1), "events_group1": int(g1["event"].sum()),
            "logrank_p": pval}


# ---------------------------------------------------------------------------
# Group meta (NPG palette: blue #3C5488, red #E64B35)
# ---------------------------------------------------------------------------
# All series (full population by model prediction)
GROUPS_RISK = {0: ("Low score",  "#E64B35"),
               1: ("High score", "#3C5488")}
# All series step 4 (true label)
GROUPS_LABEL = {0: ("1p/19q (-)", "#E64B35"),
                1: ("1p/19q (+)", "#3C5488")}
# Pos series (predicted-positive subset by TRUE label)
GROUPS_PRED_POS_TRUE = {0: ("FP (1p/19q -)", "#E64B35"),
                        1: ("TP (1p/19q +)", "#3C5488")}
# Predicted-positive subset by SCORE TIER (median split on prob)
# Same-family blues to convey dose-response (deeper = higher score = better prognosis)
GROUPS_PRED_POS_SCORE = {0: ("Low-score",  "#7BAFD4"),
                         1: ("High-score", "#1A3A6C")}

COHORTS = ["BTH", "FMUUH", "TCGA"]


def _save(fig, fig_path):
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-panel renderers
# ---------------------------------------------------------------------------

def _single_pooled(df, fig_path, group_col, group_meta,
                   legend_title="Model prediction", legend_order=None,
                   title_kind="pooled", cut_label=None):
    fig = plt.figure(figsize=(89 * MM, 90 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.0],
                          hspace=0.65, left=0.22, right=0.97,
                          top=0.90, bottom=0.10)
    ax_km = fig.add_subplot(gs[0, 0]); ax_tab = fig.add_subplot(gs[1, 0])
    n_total = len(df); n_events = int(df['event'].sum())
    if title_kind == "pred_pos_pooled":
        title = (rf"Predicted-positive subset — Pooled dataset "
                 rf"($\mathit{{n}}$ = {n_total}, events = {n_events})")
    elif title_kind == "predpos_score_pooled":
        title = (rf"Predicted-positive: score-tier in Pooled dataset "
                 rf"($\mathit{{n}}$ = {n_total}, events = {n_events})")
    elif title_kind == "predpos_cut_pooled":
        title = (rf"Predicted-positive (cutoff={cut_label}): score-tier in Pooled dataset "
                 rf"($\mathit{{n}}$ = {n_total}, events = {n_events})")
    else:
        title = (rf"Survival prediction in Pooled dataset "
                 rf"($\mathit{{n}}$ = {n_total}, events = {n_events})")
    s = km_panel(ax_km, ax_tab, df, group_col, group_meta, title=title,
                 t_step=20, t_step_risk=10,
                 legend_group_title=legend_title,
                 legend_order=legend_order)
    _save(fig, fig_path)
    return s


def _single_cohort_panel(df, fig_path, group_col, group_meta,
                         cohort, cohort_filter_col="cohort",
                         title_kind="cohort", grade_label=None, grade_val=None,
                         grade_col="grade_high",
                         legend_title="Model prediction",
                         legend_order=None, cut_label=None):
    """title_kind in {cohort, cohort_grade, pred_pos_cohort, pred_pos_cohort_grade}."""
    sub = df[df[cohort_filter_col] == cohort].copy()
    if title_kind in ("cohort_grade", "pred_pos_cohort_grade",
                      "predpos_score_cohort_grade",
                      "predpos_cut_cohort_grade"):
        sub = sub[sub[grade_col] == grade_val].copy()

    fig = plt.figure(figsize=(89 * MM, 90 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.0],
                          hspace=0.65, left=0.22, right=0.97,
                          top=0.90, bottom=0.10)
    ax_km = fig.add_subplot(gs[0, 0]); ax_tab = fig.add_subplot(gs[1, 0])

    n = len(sub); ev = int(sub["event"].sum()) if len(sub) else 0
    if title_kind == "cohort_grade":
        title = (rf"Survival prediction in {cohort} / {grade_label} "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "pred_pos_cohort":
        title = (rf"Predicted-positive subset — {cohort} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "pred_pos_cohort_grade":
        title = (rf"Predicted-positive subset — {cohort} / {grade_label} "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "predpos_score_cohort":
        title = (rf"Predicted-positive: score-tier in {cohort} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "predpos_score_cohort_grade":
        title = (rf"Predicted-positive: score-tier in {cohort} / {grade_label} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "predpos_cut_cohort":
        title = (rf"Predicted-positive (cutoff={cut_label}): score-tier in {cohort} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    elif title_kind == "predpos_cut_cohort_grade":
        title = (rf"Predicted-positive (cutoff={cut_label}): score-tier in {cohort} / {grade_label} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")
    else:
        title = (rf"Survival prediction in {cohort} dataset "
                 rf"($\mathit{{n}}$ = {n}, events = {ev})")

    if n == 0 or ev == 0:
        ax_km.text(0.5, 0.5, f"{title}\n(no events)",
                   transform=ax_km.transAxes, ha="center", va="center",
                   fontsize=7, color="grey")
        ax_km.set_xticks([]); ax_km.set_yticks([])
        for sp in ax_km.spines.values(): sp.set_visible(False)
        ax_tab.set_xticks([]); ax_tab.set_yticks([])
        for sp in ax_tab.spines.values(): sp.set_visible(False)
        _save(fig, fig_path)
        out = {"cohort": cohort,
               "n_total": n, "events_total": ev,
               "n_group0": int((sub[group_col] == 0).sum()) if n else 0,
               "events_group0": int(sub.loc[sub[group_col] == 0, "event"].sum()) if n else 0,
               "n_group1": int((sub[group_col] == 1).sum()) if n else 0,
               "events_group1": int(sub.loc[sub[group_col] == 1, "event"].sum()) if n else 0,
               "logrank_p": float("nan")}
        if title_kind in ("cohort_grade", "pred_pos_cohort_grade",
                          "predpos_score_cohort_grade",
                          "predpos_cut_cohort_grade"):
            out["grade"] = grade_label
        return out

    s = km_panel(ax_km, ax_tab, sub, group_col, group_meta,
                 title=title, title_fontsize=7,
                 legend_group_title=legend_title,
                 legend_order=legend_order)
    _save(fig, fig_path)
    out = {"cohort": cohort, **s}
    if title_kind in ("cohort_grade", "pred_pos_cohort_grade",
                      "predpos_score_cohort_grade",
                      "predpos_cut_cohort_grade"):
        out["grade"] = grade_label
    return out


# ============================================================
# Load full cohort table once — shared by PART A & PART B
# ============================================================
df_all = pd.read_csv(DATA_ALL)
print(f"loaded {len(df_all)} slides, events={int(df_all['event'].sum())}, "
      f"cohort×label_1p19q breakdown:")
print(df_all.groupby(["cohort", "label_1p19q"]).agg(
    slides=("slide_id", "size"), events=("event", "sum")).to_string())

# ============================================================
# PART A — PREDICTED-POSITIVE subset
#   risk_high_typ == 1 subset of the full cohort,
#   grouped by TRUE label (label_1p19q): TP vs FP
# ============================================================
print("\n" + "=" * 60)
print("PART A — Predicted-positive subset (TP vs FP)")
print("=" * 60)
df_pred_pos = df_all[df_all["risk_high_typ"] == 1].copy()
print(f"predicted-positive subset: {len(df_pred_pos)} slides, "
      f"events={int(df_pred_pos['event'].sum())}, "
      f"cohort breakdown={dict(df_pred_pos['cohort'].value_counts())}")
print(df_pred_pos.groupby(["cohort", "label_1p19q"]).agg(
    slides=("slide_id", "size"), events=("event", "sum")).to_string())

POS_LEGEND_TITLE = "True label"
POS_LEGEND_ORDER = ["TP (1p/19q +)", "FP (1p/19q -)"]

# Step 1 — pooled predicted-pos by true label (TP vs FP)
s1p = _single_pooled(df_pred_pos,
                     FIG / "km_pos_step1_pooled_TPvsFP.pdf",
                     "label_1p19q", GROUPS_PRED_POS_TRUE,
                     legend_title=POS_LEGEND_TITLE,
                     legend_order=POS_LEGEND_ORDER,
                     title_kind="pred_pos_pooled")
print(f"[pos/step1] n={s1p['n_total']} events={s1p['events_total']} "
      f"P={s1p['logrank_p']:.4f}")

# Step 2 — per-cohort predicted-pos by true label
log2p = []
for cohort in COHORTS:
    s = _single_cohort_panel(df_pred_pos,
            FIG / f"km_pos_step2_{cohort}.pdf",
            "label_1p19q", GROUPS_PRED_POS_TRUE,
            cohort=cohort, title_kind="pred_pos_cohort",
            legend_title=POS_LEGEND_TITLE,
            legend_order=POS_LEGEND_ORDER)
    log2p.append(s)
    print(f"[pos/step2/{cohort}] n={s['n_total']} events={s['events_total']} "
          f"P={s['logrank_p']:.4f}")

# Step 3 — per-cohort × grade predicted-pos by true label
GRADE_STRATA = [(0, "Grade 2", "G2"), (1, "Grade >=3", "Ge3")]
log3p = []
for cohort in COHORTS:
    for gv, glab, gtag in GRADE_STRATA:
        s = _single_cohort_panel(df_pred_pos,
                FIG / f"km_pos_step3_{cohort}_{gtag}.pdf",
                "label_1p19q", GROUPS_PRED_POS_TRUE,
                cohort=cohort, title_kind="pred_pos_cohort_grade",
                grade_label=glab, grade_val=gv,
                legend_title=POS_LEGEND_TITLE,
                legend_order=POS_LEGEND_ORDER)
        log3p.append(s)
        print(f"[pos/step3/{cohort}/{glab}] n={s['n_total']} ev={s['events_total']} "
              f"FP/TP={s['n_group0']}/{s['n_group1']} "
              f"ev FP/TP={s['events_group0']}/{s['events_group1']} P={s['logrank_p']:.4f}")


# ============================================================
# PART C — Predicted-positive subset, SCORE-TIER stratification
#   Method (per-cohort median): for each cohort, compute the median
#   `prob` within that cohort's predicted-positive subset and use it
#   as a cohort-specific cutoff. Tier 1 = High-score (prob > cohort
#   median), Tier 0 = Low-score (prob <= cohort median, but still
#   > Youden cutoff because subset already passed risk_high_typ).
#
#   Rationale: pooled median is contaminated by cohort domain shift
#   (different scanners / staining), which previously dragged TCGA
#   into a near-null split. Splitting within each cohort avoids that.
#   Pooled KM simply concatenates the per-cohort-split frames.
#
#   Clinical question: within model-positive cases, does a higher
#   model probability translate into BETTER prognosis (dose-response
#   support for prob as a continuous severity/typicality indicator,
#   not just a binary marker)?
# ============================================================
print("\n" + "=" * 60)
print("PART C — Predicted-positive subset, score-tier KM "
      "(per-cohort median cutoff)")
print("=" * 60)

df_pred_pos_score = df_pred_pos.copy()
df_pred_pos_score["score_tier"] = 0
medians_per_cohort = {}
for c in COHORTS:
    mask = df_pred_pos_score["cohort"] == c
    if not mask.any():
        medians_per_cohort[c] = float("nan")
        continue
    med_c = float(df_pred_pos_score.loc[mask, "prob"].median())
    medians_per_cohort[c] = med_c
    df_pred_pos_score.loc[mask & (df_pred_pos_score["prob"] > med_c),
                          "score_tier"] = 1
    n_high = int((mask & (df_pred_pos_score["score_tier"] == 1)).sum())
    n_low = int((mask & (df_pred_pos_score["score_tier"] == 0)).sum())
    print(f"[predpos median] {c}: median={med_c:.4f}, "
          f"n_high={n_high}, n_low={n_low}")

# Backwards-compatible scalar (used downstream only as a reference in the
# log CSV); kept as the pooled median for sanity but no longer used to split.
median_pos = float(df_pred_pos["prob"].median())
print(f"(reference) pooled predicted-positive median(prob) = {median_pos:.6f}")
print(f"min/max(prob) on predicted-pos subset: "
      f"{df_pred_pos['prob'].min():.4f} / {df_pred_pos['prob'].max():.4f}")
print("score_tier counts (1=High, 0=Low):",
      dict(df_pred_pos_score["score_tier"].value_counts()))
print(df_pred_pos_score.groupby(["cohort", "score_tier"]).agg(
    slides=("slide_id", "size"), events=("event", "sum")).to_string())

PREDPOS_SCORE_LEGEND_TITLE = "Score tier"
PREDPOS_SCORE_LEGEND_ORDER = ["High-score", "Low-score"]

# Step 1c — pooled predicted-pos by score tier
s1c = _single_pooled(df_pred_pos_score,
                     FIG / "km_predpos_score_step1_pooled.pdf",
                     "score_tier", GROUPS_PRED_POS_SCORE,
                     legend_title=PREDPOS_SCORE_LEGEND_TITLE,
                     legend_order=PREDPOS_SCORE_LEGEND_ORDER,
                     title_kind="predpos_score_pooled")
print(f"[predpos_score/step1] n={s1c['n_total']} events={s1c['events_total']} "
      f"P={s1c['logrank_p']:.4f}")

# Step 2c — per-cohort predicted-pos by score tier
log2c = []
for cohort in COHORTS:
    s = _single_cohort_panel(df_pred_pos_score,
            FIG / f"km_predpos_score_step2_{cohort}.pdf",
            "score_tier", GROUPS_PRED_POS_SCORE,
            cohort=cohort, title_kind="predpos_score_cohort",
            legend_title=PREDPOS_SCORE_LEGEND_TITLE,
            legend_order=PREDPOS_SCORE_LEGEND_ORDER)
    log2c.append(s)
    print(f"[predpos_score/step2/{cohort}] n={s['n_total']} ev={s['events_total']} "
          f"Low/High={s['n_group0']}/{s['n_group1']} "
          f"ev Low/High={s['events_group0']}/{s['events_group1']} "
          f"P={s['logrank_p']:.4f}")

# Step 3c — per-cohort × grade predicted-pos by score tier
log3c = []
for cohort in COHORTS:
    for gv, glab, gtag in GRADE_STRATA:
        s = _single_cohort_panel(df_pred_pos_score,
                FIG / f"km_predpos_score_step3_{cohort}_{gtag}.pdf",
                "score_tier", GROUPS_PRED_POS_SCORE,
                cohort=cohort, title_kind="predpos_score_cohort_grade",
                grade_label=glab, grade_val=gv,
                legend_title=PREDPOS_SCORE_LEGEND_TITLE,
                legend_order=PREDPOS_SCORE_LEGEND_ORDER)
        log3c.append(s)
        print(f"[predpos_score/step3/{cohort}/{glab}] n={s['n_total']} ev={s['events_total']} "
              f"Low/High={s['n_group0']}/{s['n_group1']} "
              f"ev Low/High={s['events_group0']}/{s['events_group1']} "
              f"P={s['logrank_p']:.4f}")


# ============================================================
# PART D / PART E — Predicted-positive, FIXED-CUTOFF score-tier KM
#   Method: within the predicted-positive subset (prob > Youden=0.5786),
#   split into High-score (prob > CUT) vs Low-score (Youden < prob <= CUT)
#   using a single fixed cutoff applied across all cohorts.
#
#   PART D: cutoff = 0.85   -> figures/predpos_cutoff_085/
#   PART E: cutoff = 0.95   -> figures/predpos_cutoff_095/
#
#   Rationale: per-cohort median (PART C) lets cohort scale drag the
#   threshold; a fixed prob threshold answers "does an above-X model
#   confidence translate into better prognosis within model-positives?"
# ============================================================

PREDPOS_CUT_LEGEND_TITLE = "Score tier"


def _run_fixed_cutoff_part(cut_value, tag, outdir, log_label):
    """Run pooled + per-cohort + per-cohort x grade KM at a single fixed
    score cutoff. Returns (pooled_summary, log2_list, log3_list)."""
    cut_str = f"{cut_value:.2f}"
    print("\n" + "=" * 60)
    print(f"{log_label} — Predicted-positive, fixed score cutoff = {cut_str}")
    print("=" * 60)

    df_cut = df_pred_pos.copy()
    df_cut["score_tier"] = (df_cut["prob"] > cut_value).astype(int)

    n_high = int((df_cut["score_tier"] == 1).sum())
    n_low = int((df_cut["score_tier"] == 0).sum())
    print(f"[{log_label}] pooled: n_high(>{cut_str})={n_high}, "
          f"n_low(<={cut_str})={n_low}")
    print(df_cut.groupby(["cohort", "score_tier"]).agg(
        slides=("slide_id", "size"), events=("event", "sum")).to_string())

    high_label = f"High-score (>{cut_str})"
    low_label = f"Low-score (≤{cut_str})"
    group_meta = {0: (low_label, "#7BAFD4"),
                  1: (high_label, "#1A3A6C")}
    legend_order = [high_label, low_label]

    # Step 1 — pooled
    s1 = _single_pooled(df_cut,
                        outdir / f"km_predpos_cut{tag}_step1_pooled.pdf",
                        "score_tier", group_meta,
                        legend_title=PREDPOS_CUT_LEGEND_TITLE,
                        legend_order=legend_order,
                        title_kind="predpos_cut_pooled",
                        cut_label=cut_str)
    print(f"[{log_label}/step1] n={s1['n_total']} events={s1['events_total']} "
          f"Low/High={s1['n_group0']}/{s1['n_group1']} "
          f"P={s1['logrank_p']:.4f}")

    # Step 2 — per cohort
    log2 = []
    for cohort in COHORTS:
        s = _single_cohort_panel(
            df_cut,
            outdir / f"km_predpos_cut{tag}_step2_{cohort}.pdf",
            "score_tier", group_meta,
            cohort=cohort, title_kind="predpos_cut_cohort",
            legend_title=PREDPOS_CUT_LEGEND_TITLE,
            legend_order=legend_order,
            cut_label=cut_str)
        log2.append(s)
        print(f"[{log_label}/step2/{cohort}] n={s['n_total']} "
              f"ev={s['events_total']} Low/High={s['n_group0']}/{s['n_group1']} "
              f"ev Low/High={s['events_group0']}/{s['events_group1']} "
              f"P={s['logrank_p']:.4f}")

    # Step 3 — per cohort x grade
    log3 = []
    for cohort in COHORTS:
        for gv, glab, gtag in GRADE_STRATA:
            s = _single_cohort_panel(
                df_cut,
                outdir / f"km_predpos_cut{tag}_step3_{cohort}_{gtag}.pdf",
                "score_tier", group_meta,
                cohort=cohort, title_kind="predpos_cut_cohort_grade",
                grade_label=glab, grade_val=gv,
                legend_title=PREDPOS_CUT_LEGEND_TITLE,
                legend_order=legend_order,
                cut_label=cut_str)
            log3.append(s)
            print(f"[{log_label}/step3/{cohort}/{glab}] n={s['n_total']} "
                  f"ev={s['events_total']} "
                  f"Low/High={s['n_group0']}/{s['n_group1']} "
                  f"ev Low/High={s['events_group0']}/{s['events_group1']} "
                  f"P={s['logrank_p']:.4f}")

    return s1, log2, log3


# PART D — cutoff 0.85
s1d, log2d, log3d = _run_fixed_cutoff_part(
    cut_value=0.85, tag="085", outdir=FIG_CUT085, log_label="predpos_cut085")

# PART E — cutoff 0.95
s1e, log2e, log3e = _run_fixed_cutoff_part(
    cut_value=0.95, tag="095", outdir=FIG_CUT095, log_label="predpos_cut095")


# ============================================================
# PART B — FULL (+ AND -) Step 1/2/3 + Step 4 (by 1p19q label)
#   Unchanged from round 5 (no mOS annotations).
# ============================================================
print("\n" + "=" * 60)
print("PART B — Full cohort (by model prediction; step 4 by true label)")
print("=" * 60)

# Step 1' — pooled all by model
s1a = _single_pooled(df_all, FIG / "km_all_step1_pooled_model.pdf",
                     "risk_high_typ", GROUPS_RISK)
print(f"[all/step1] n={s1a['n_total']} events={s1a['events_total']} "
      f"P={s1a['logrank_p']:.4f}")

# Step 2' — per-cohort all by model
log2a = []
for cohort in COHORTS:
    s = _single_cohort_panel(df_all,
            FIG / f"km_all_step2_{cohort}.pdf",
            "risk_high_typ", GROUPS_RISK,
            cohort=cohort, title_kind="cohort")
    log2a.append(s)
    print(f"[all/step2/{cohort}] n={s['n_total']} events={s['events_total']} "
          f"P={s['logrank_p']:.4f}")

# Step 3' — per-cohort × grade by model
log3a = []
for cohort in COHORTS:
    for gv, glab, gtag in GRADE_STRATA:
        s = _single_cohort_panel(df_all,
                FIG / f"km_all_step3_{cohort}_{gtag}.pdf",
                "risk_high_typ", GROUPS_RISK,
                cohort=cohort, title_kind="cohort_grade",
                grade_label=glab, grade_val=gv)
        log3a.append(s)
        print(f"[all/step3/{cohort}/{glab}] n={s['n_total']} ev={s['events_total']} "
              f"high/low={s['n_group1']}/{s['n_group0']} "
              f"P={s['logrank_p']:.4f}")

# Step 4' — per-cohort by 1p/19q label itself (sanity)
log4a = []
for cohort in COHORTS:
    s = _single_cohort_panel(df_all,
            FIG / f"km_all_step4_{cohort}.pdf",
            "label_1p19q", GROUPS_LABEL,
            cohort=cohort, title_kind="cohort",
            legend_title="True label")
    log4a.append(s)
    print(f"[all/step4/{cohort}] n={s['n_total']} events={s['events_total']} "
          f"P={s['logrank_p']:.4f}")

# Consolidated CSV
rows = []
rows.append({"part": "pos", "step": 1, "cohort": "POOLED", **s1p})
for s in log2p: rows.append({"part": "pos", "step": 2, **s})
for s in log3p: rows.append({"part": "pos", "step": 3, **s})
rows.append({"part": "all", "step": 1, "cohort": "POOLED", **s1a})
for s in log2a: rows.append({"part": "all", "step": 2, **s})
for s in log3a: rows.append({"part": "all", "step": 3, **s})
for s in log4a: rows.append({"part": "all", "step": 4, **s})
rows.append({"part": "predpos_score", "step": 1, "cohort": "POOLED",
             "median_prob": median_pos, **s1c})
for s in log2c:
    coh = s.get("cohort")
    rows.append({"part": "predpos_score", "step": 2,
                 "median_prob": medians_per_cohort.get(coh, float("nan")),
                 **s})
for s in log3c:
    coh = s.get("cohort")
    rows.append({"part": "predpos_score", "step": 3,
                 "median_prob": medians_per_cohort.get(coh, float("nan")),
                 **s})

# PART D — fixed cutoff 0.85
rows.append({"part": "predpos_cut085", "step": 1, "cohort": "POOLED",
             "cutoff": 0.85, **s1d})
for s in log2d:
    rows.append({"part": "predpos_cut085", "step": 2, "cutoff": 0.85, **s})
for s in log3d:
    rows.append({"part": "predpos_cut085", "step": 3, "cutoff": 0.85, **s})

# PART E — fixed cutoff 0.95
rows.append({"part": "predpos_cut095", "step": 1, "cohort": "POOLED",
             "cutoff": 0.95, **s1e})
for s in log2e:
    rows.append({"part": "predpos_cut095", "step": 2, "cutoff": 0.95, **s})
for s in log3e:
    rows.append({"part": "predpos_cut095", "step": 3, "cutoff": 0.95, **s})

out_csv = RES / "km_three_steps_logrank.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nwrote {out_csv}")
