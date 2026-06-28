"""Multivariate Cox regression.

Model: Hazard ~ prob + Age + WHO_grade_high

NOTE: Resection extent is NOT included — none of the three cohorts
(BTH / FMUUH / TCGA) provide a resection-extent field in the canonical
cohort_table_all.csv.

Four independent Cox models:
  - BTH only
  - FMUUH only
  - TCGA only
  - Pooled (3 cohorts; stratified by cohort)

All slides are treated as independent observations (no patient cluster
correction), matching the KM convention.

Output:
  - results/multivariate_cox.csv  (cohort, variable, HR, lo95, hi95, P)
  - figures/cox_forest_multivariate.{pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parent
DATA_ALL = ROOT / "data/cohort_table_all.csv"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

# ---- Nature-grade rcParams (kept identical to KM script) -----------------
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

MM = 1 / 25.4


def _fmt_p(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Load + prepare
# ---------------------------------------------------------------------------
df_all = pd.read_csv(DATA_ALL)

REQUIRED = ["cohort", "OS_months", "event", "prob", "Age", "grade_high"]
missing = [c for c in REQUIRED if c not in df_all.columns]
if missing:
    raise SystemExit(f"missing required columns: {missing}")

# Keep only rows with all 3 covariates + survival
df = df_all.dropna(subset=["OS_months", "event", "prob", "Age", "grade_high"]).copy()
# strict positive duration
df = df[df["OS_months"] > 0].copy()
print(f"complete-case slides: {len(df)} (of {len(df_all)})  events={int(df['event'].sum())}")

COVARS = ["prob", "Age", "grade_high"]
COVAR_LABEL = {"prob": "Model probability",
               "Age": "Age",
               "grade_high": "WHO grade (>=3)"}


def _fit_cox(sub, stratify_cohort=False):
    cols = COVARS + ["OS_months", "event"]
    if stratify_cohort:
        sub2 = sub[cols + ["cohort"]].copy()
        cph = CoxPHFitter()
        cph.fit(sub2, duration_col="OS_months", event_col="event",
                strata=["cohort"])
    else:
        sub2 = sub[cols].copy()
        cph = CoxPHFitter()
        cph.fit(sub2, duration_col="OS_months", event_col="event")
    return cph


COHORT_RUNS = [
    ("BTH",   df[df["cohort"] == "BTH"].copy(),   False),
    ("FMUUH", df[df["cohort"] == "FMUUH"].copy(), False),
    ("TCGA",  df[df["cohort"] == "TCGA"].copy(),  False),
    ("Pooled (cohort-stratified)", df.copy(),     True),
]

rows = []
for cohort_name, sub, stratify in COHORT_RUNS:
    n = len(sub); ev = int(sub["event"].sum())
    print(f"\n=== {cohort_name}: n={n}, events={ev} ===")
    if n < 10 or ev < 5:
        for var in COVARS:
            rows.append({"cohort": cohort_name, "n": n, "events": ev,
                         "variable": var, "HR": float("nan"),
                         "lo95": float("nan"), "hi95": float("nan"),
                         "P": float("nan")})
        print("  insufficient data — skipped")
        continue
    try:
        cph = _fit_cox(sub, stratify_cohort=stratify)
    except Exception as e:
        print(f"  fit failed: {e}")
        for var in COVARS:
            rows.append({"cohort": cohort_name, "n": n, "events": ev,
                         "variable": var, "HR": float("nan"),
                         "lo95": float("nan"), "hi95": float("nan"),
                         "P": float("nan")})
        continue
    summ = cph.summary
    print(summ[["exp(coef)", "exp(coef) lower 95%",
                "exp(coef) upper 95%", "p"]].round(4).to_string())
    for var in COVARS:
        if var in summ.index:
            rows.append({"cohort": cohort_name, "n": n, "events": ev,
                         "variable": var,
                         "HR": float(summ.loc[var, "exp(coef)"]),
                         "lo95": float(summ.loc[var, "exp(coef) lower 95%"]),
                         "hi95": float(summ.loc[var, "exp(coef) upper 95%"]),
                         "P": float(summ.loc[var, "p"])})
        else:
            rows.append({"cohort": cohort_name, "n": n, "events": ev,
                         "variable": var, "HR": float("nan"),
                         "lo95": float("nan"), "hi95": float("nan"),
                         "P": float("nan")})

df_out = pd.DataFrame(rows, columns=["cohort", "n", "events", "variable",
                                     "HR", "lo95", "hi95", "P"])
out_csv = RES / "multivariate_cox.csv"
df_out.to_csv(out_csv, index=False)
print(f"\nwrote {out_csv}")


# ---------------------------------------------------------------------------
# Forest plot — grouped by variable (3 var × 4 cohort = 12 rows)
# ---------------------------------------------------------------------------
COHORT_ORDER = ["BTH", "FMUUH", "TCGA", "Pooled (cohort-stratified)"]
VAR_ORDER = ["prob", "Age", "grade_high"]
NPG_BLUE = "#3C5488"
GREY = "#888888"

# Row layout: variable groups separated by a blank row
rows_plot = []  # list of (y, kind, payload)
y = 0
group_spacing = 1
for vi, var in enumerate(VAR_ORDER):
    rows_plot.append((y, "header", var))
    y += 1
    for ch in COHORT_ORDER:
        rec = df_out[(df_out["variable"] == var)
                     & (df_out["cohort"] == ch)]
        if len(rec) == 0:
            continue
        rows_plot.append((y, "row", rec.iloc[0].to_dict()))
        y += 1
    y += group_spacing  # blank gap between variable groups

n_rows = y
fig_h_mm = 12 + 6 * n_rows
fig = plt.figure(figsize=(183 * MM, fig_h_mm * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0],
                      wspace=0.04, left=0.02, right=0.98,
                      top=0.96, bottom=0.10)
ax_text = fig.add_subplot(gs[0, 0])
ax_for = fig.add_subplot(gs[0, 1])

ax_text.set_xlim(0, 1); ax_text.set_ylim(-0.5, n_rows - 0.5)
ax_text.invert_yaxis()
ax_text.set_xticks([]); ax_text.set_yticks([])
for sp in ax_text.spines.values(): sp.set_visible(False)

ax_for.set_ylim(-0.5, n_rows - 0.5)
ax_for.invert_yaxis()
ax_for.set_yticks([])
ax_for.spines["left"].set_visible(False)

# Determine HR x-range
hrs_all = df_out["HR"].dropna().values
los_all = df_out["lo95"].dropna().values
his_all = df_out["hi95"].dropna().values
candidates = np.concatenate([hrs_all, los_all, his_all]) if len(hrs_all) else np.array([0.5, 2.0])
candidates = candidates[(candidates > 1e-3) & np.isfinite(candidates)]
x_min = max(1e-2, float(np.nanmin(candidates)) * 0.7) if len(candidates) else 0.1
x_max = float(np.nanmax(candidates)) * 1.2 if len(candidates) else 10.0
# clamp to sane bounds for log axis
x_min = max(0.05, min(x_min, 0.5))
x_max = min(50.0, max(x_max, 5.0))
ax_for.set_xscale("log")
ax_for.set_xlim(x_min, x_max)

# Reference line at HR=1
ax_for.axvline(1.0, color=GREY, lw=0.5, linestyle="-")

# Text column layout (axes fraction). Right-aligned numeric columns.
COL_N    = 0.40
COL_EV   = 0.50
COL_HR   = 0.88   # right edge of HR (95% CI)
COL_P    = 1.00   # right edge of P

# Text column headers
ax_text.text(0.00, -0.5, "Variable / Cohort", fontsize=6.5,
             weight="bold", ha="left", va="center")
ax_text.text(COL_N, -0.5, "n", fontsize=6.5, weight="bold",
             ha="right", va="center")
ax_text.text(COL_EV, -0.5, "events", fontsize=6.5, weight="bold",
             ha="right", va="center")
ax_text.text(COL_HR, -0.5, "HR (95% CI)", fontsize=6.5,
             weight="bold", ha="right", va="center")
ax_text.text(COL_P, -0.5, "P", fontsize=6.5, weight="bold",
             ha="right", va="center")

for y_pos, kind, payload in rows_plot:
    if kind == "header":
        ax_text.text(0.00, y_pos, COVAR_LABEL[payload],
                     fontsize=7, weight="bold",
                     ha="left", va="center", color="#222222")
        # no forest mark on header row
        continue
    rec = payload
    cohort = rec["cohort"]
    var = rec["variable"]
    hr = rec["HR"]; lo = rec["lo95"]; hi = rec["hi95"]; p = rec["P"]
    n_v = rec["n"]; ev_v = rec["events"]

    # Left text panel
    ax_text.text(0.03, y_pos, cohort, fontsize=6.5,
                 ha="left", va="center", color="#222222")
    ax_text.text(COL_N, y_pos, f"{int(n_v)}" if pd.notna(n_v) else "—",
                 fontsize=6.5, ha="right", va="center")
    ax_text.text(COL_EV, y_pos, f"{int(ev_v)}" if pd.notna(ev_v) else "—",
                 fontsize=6.5, ha="right", va="center")
    if pd.notna(hr) and pd.notna(lo) and pd.notna(hi):
        hr_str = f"{hr:.2f} ({lo:.2f}-{hi:.2f})"
    else:
        hr_str = "—"
    p_str = _fmt_p(p)
    ax_text.text(COL_HR, y_pos, hr_str,
                 fontsize=6.5, ha="right", va="center")
    ax_text.text(COL_P, y_pos, p_str,
                 fontsize=6.5, ha="right", va="center")

    # Right forest panel
    if pd.notna(hr) and pd.notna(lo) and pd.notna(hi) and hr > 0:
        lo_c = max(lo, x_min); hi_c = min(hi, x_max)
        ax_for.plot([lo_c, hi_c], [y_pos, y_pos],
                    color=GREY, lw=0.8, solid_capstyle="butt", zorder=2)
        ax_for.plot([hr], [y_pos], "o", color=NPG_BLUE,
                    markersize=3.5, markeredgewidth=0, zorder=3)

ax_for.set_xlabel("Hazard ratio (log scale)", labelpad=2)
ax_for.tick_params(axis="x", pad=2)
ax_for.tick_params(axis="y", left=False)

fig.suptitle("Multivariate Cox: Hazard ~ Model probability + Age + WHO grade",
             fontsize=8, weight="bold", y=0.995)

out_pdf = FIG / "cox_forest_multivariate.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}")
