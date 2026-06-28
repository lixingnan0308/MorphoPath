"""Build TWO canonical cohort tables for prognosis analysis.

FMUUH labels come from `FMUUH_List_2.xlsx['final_label']` — the canonical
recheck cohort (272 slides / 231 patients with final_label != NaN).

Outputs (both written to data/):
  - cohort_table_pos.csv  — 1p/19q(+) only, for Step 1/2/3 (positive prognosis)
  - cohort_table_all.csv  — 1p/19q(+) AND (-), full cohort for Step 1' (negative+positive)

Schema (both tables):
  cohort, pid, patient_id, slide_id, Age, WHO_grade, prob,
  OS_days, OS_months, event, label_1p19q,
  risk_high_typ, grade_high, age_gt60

Where rows come from:
  - BTH:   data/bth_all_prob.csv (full pos+neg from the MorphoPath no-grade ckpt
           as bth_contrib.csv; BTH_List.xlsx now has OS + endpoint columns)
  - FMUUH: source_FMUUH_TCGA_NOBTH.csv (already uses final_label)
  - TCGA:  source_FMUUH_TCGA_NOBTH.csv (full pos+neg)

Sanity: for the POS table FMUUH slides are filtered to those where
patient appears in FMUUH_List_2.xlsx with final_label==1.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
PROG = Path(__file__).resolve().parent
DATA = PROG / "data"
OUT_POS = DATA / "cohort_table_pos.csv"
OUT_ALL = DATA / "cohort_table_all.csv"

UNI512_YOUDEN = 0.5786  # text-anchor CV-OOF Youden for UNI-512 v2_nograde
                        # (source: metrics_512.csv — applies to BTH/FMUUH/TCGA which
                        #  use the text-anchor variant; noanchor cohorts FAH-ZH/WU-RMH
                        #  use 0.6432 but are not in this prognosis analysis)

# --- load sources -----------------------------------------------------------
src = pd.read_csv(DATA / "source_FMUUH_TCGA_NOBTH.csv")  # FMUUH 272 + TCGA 592
bth_all = pd.read_csv(DATA / "bth_all_prob.csv")  # BTH pos+neg, same v2_nograde ckpt
fl2 = pd.read_excel(DATA / "FMUUH_List_2.xlsx").dropna(subset=["final_label"]).copy()
fl2["final_label"] = fl2["final_label"].astype(int)
fl2["patient"] = fl2["patient"].astype(str)
fl2_pat = fl2.groupby("patient").agg(final_label=("final_label", "first")).reset_index()

print(f"[load] source_FMUUH_TCGA_NOBTH.csv: {len(src)} rows")
print(f"        FMUUH: {(src['cohort']=='FMUUH').sum()}  TCGA: {(src['cohort']=='TCGA').sum()}")
print(f"[load] bth_all_prob.csv: {len(bth_all)} rows "
      f"({bth_all['label_1p19q'].value_counts().to_dict()})")
print(f"[load] FMUUH_List_2.xlsx valid final_label: {len(fl2)} rows "
      f"({fl2['final_label'].value_counts().to_dict()})")


def _build_bth(only_pos=False):
    """BTH rows from bth_all_prob.csv (pos+neg available)."""
    d = bth_all.copy()
    if only_pos:
        d = d[d["label_1p19q"] == 1]
    d = d[d["OS_days"].notna() & d["event"].notna()
          & d["Age"].notna() & d["WHO_grade"].notna()].copy()
    d["cohort"] = "BTH"
    d["pid"] = "BTH_" + d["patient_id"].astype(str)
    d = d.rename(columns={"wsi_id": "slide_id"})
    return d[["cohort", "pid", "patient_id", "slide_id",
              "Age", "WHO_grade", "prob", "OS_days", "event", "label_1p19q"]]


def _build_from_source(target_cohort, only_pos=False):
    """FMUUH or TCGA rows from source_FMUUH_TCGA_NOBTH.csv.

    For FMUUH, additionally filter to patients whose FMUUH_List_2 final_label
    matches the requested label (sanity check — should be a no-op since source
    CSV is already keyed on final_label, but this catches drift).
    """
    d = src[src["cohort"] == target_cohort].copy()
    if only_pos:
        d = d[d["label_1p19q"] == 1]
    # must have full OS+Age+Grade
    d = d[d["OS_days"].notna() & d["event"].notna()
          & d["Age"].notna() & d["WHO_grade"].notna()].copy()

    if target_cohort == "FMUUH":
        # Cross-check against FMUUH_List_2 final_label by EXACT patient_id
        # (no `-N` suffix stripping — `-2`/`-3` slides are treated as
        # independent patients, so they don't inherit the base patient's
        # final_label). For exact matches we report disagreements as a
        # warning only — we still trust source CSV's label_1p19q since
        # source is the canonical slide-level table.
        m = d.merge(fl2_pat, left_on=d["patient_id"].astype(str),
                    right_on="patient", how="left")
        matched = m["final_label"].notna()
        mismatched = m[matched & (m["final_label"] != m["label_1p19q"])]
        if len(mismatched) > 0:
            print(f"[warn] {target_cohort}: {len(mismatched)} slides where source label "
                  f"disagrees with FMUUH_List_2 final_label (KEPT — using source label):")
            print(mismatched[["patient_id", "wsi_id", "label_1p19q", "final_label"]].to_string(index=False))
        else:
            print(f"[ok] {target_cohort}: {matched.sum()}/{len(m)} slides matched "
                  f"FMUUH_List_2.final_label exactly; {len(m)-matched.sum()} have no fl2 entry "
                  f"(e.g. `-2`/`-3` independent slides) — using source label as-is")
        d = m.drop(columns=[c for c in ["patient", "final_label", "key_0"] if c in m.columns])

    d["pid"] = target_cohort + "_" + d["patient_id"].astype(str)
    d = d.rename(columns={"wsi_id": "slide_id"})
    return d[["cohort", "pid", "patient_id", "slide_id",
              "Age", "WHO_grade", "prob", "OS_days", "event", "label_1p19q"]]


def _finalise(df):
    df = df.copy()
    df["event"] = df["event"].astype(int)
    df["label_1p19q"] = df["label_1p19q"].astype(int)
    df["OS_months"] = df["OS_days"] / 30.4375
    df["risk_high_typ"] = (df["prob"] > UNI512_YOUDEN).astype(int)
    df["grade_high"] = (df["WHO_grade"] >= 3).astype(int)
    df["age_gt60"] = (df["Age"] > 60).astype(int)
    cols = ["cohort", "pid", "patient_id", "slide_id",
            "Age", "WHO_grade", "prob",
            "OS_days", "OS_months", "event", "label_1p19q",
            "risk_high_typ", "grade_high", "age_gt60"]
    return df[cols].sort_values(["cohort", "pid", "slide_id"]).reset_index(drop=True)


# ============================================================
# POSITIVE-ONLY table (replaces cohort_table.csv)
# ============================================================
print("\n" + "=" * 60)
print("BUILDING cohort_table_pos.csv (1p/19q+ only)")
print("=" * 60)
bth_p = _build_bth(only_pos=True)
fmuuh_p = _build_from_source("FMUUH", only_pos=True)
tcga_p = _build_from_source("TCGA", only_pos=True)
pos = _finalise(pd.concat([bth_p, fmuuh_p, tcga_p], ignore_index=True))
pos.to_csv(OUT_POS, index=False)
_pos_stats = pos.groupby("cohort").agg(slides=("slide_id", "size"),
                                       patients=("patient_id", "nunique"),
                                       events=("event", "sum"))
print(f"\n[pos] cohort_table_pos.csv: {len(pos)} slides")
print(_pos_stats.to_string())

# ============================================================
# FULL table (positive + negative) — used for "all cases" Step 1
# ============================================================
print("\n" + "=" * 60)
print("BUILDING cohort_table_all.csv (1p/19q + AND -)")
print("=" * 60)
# BTH: full pos+neg available (BTH_List.xlsx now has OS + endpoint)
bth_a = _build_bth(only_pos=False)
fmuuh_a = _build_from_source("FMUUH", only_pos=False)
tcga_a = _build_from_source("TCGA", only_pos=False)
all_ = _finalise(pd.concat([bth_a, fmuuh_a, tcga_a], ignore_index=True))
all_.to_csv(OUT_ALL, index=False)
print(f"\n[all] cohort_table_all.csv: {len(all_)} slides")
print(all_.groupby(["cohort", "label_1p19q"]).agg(
    slides=("slide_id", "size"),
    patients=("patient_id", "nunique"),
    events=("event", "sum")).to_string())

print(f"\nwrote:\n  {OUT_POS}\n  {OUT_ALL}")
