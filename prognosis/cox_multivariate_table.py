"""Multivariate Cox table (with Sex) as a booktabs PDF, formatted like
visual4zeya/tables/table_pdf_version/uni_512.pdf.

Model: Hazard ~ Model probability + Age + grade + Sex
Cohorts: BTH / FMUUH / TCGA  (NO pooled row).
Variables are sub-table headers (band rows); cohorts are the rows beneath.
Columns: Cohort | n | events | HR (95% CI) | P
Title (caption): "Multivariate Cox analysis".

Sex coding: 1 = Male, 0 = Female (verified: source_FMUUH_TCGA Sex 1<->Male;
BTH_List M->1, F->0). HR is Male vs Female.

Outputs: results/multivariate_cox_sex.csv, results/multivariate_cox_table.{tex,pdf}
"""
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"

# --- load + merge Sex -------------------------------------------------------
ct = pd.read_csv(DATA / "cohort_table_all.csv")
src = pd.read_csv(DATA / "source_FMUUH_TCGA_NOBTH.csv")
bl = pd.read_excel(DATA / "BTH_List.xlsx")[["ID", "Sex"]].copy()
bl["ID"] = bl["ID"].astype(str)
bl["sex_male"] = bl["Sex"].map({"M": 1, "F": 0})

ct["patient_id"] = ct["patient_id"].astype(str)
ct["slide_id"] = ct["slide_id"].astype(str)

# BTH Sex by patient_id <-> BTH_List ID
bth = ct[ct.cohort == "BTH"].merge(
    bl[["ID", "sex_male"]].drop_duplicates("ID"),
    left_on="patient_id", right_on="ID", how="left").drop(columns="ID")

# FMUUH / TCGA Sex by slide_id <-> source wsi_id  (source Sex already 1=Male)
ext = ct[ct.cohort.isin(["FMUUH", "TCGA"])].copy()
s = src[["cohort", "wsi_id", "Sex"]].copy()
s["wsi_id"] = s["wsi_id"].astype(str)
ext = ext.merge(s.drop_duplicates(["cohort", "wsi_id"]),
                left_on=["cohort", "slide_id"], right_on=["cohort", "wsi_id"],
                how="left").rename(columns={"Sex": "sex_male"}).drop(columns="wsi_id")

df = pd.concat([bth, ext], ignore_index=True)
df = df.dropna(subset=["OS_months", "event", "prob", "Age", "grade_high", "sex_male"])
df = df[df["OS_months"] > 0].copy()
df["sex_male"] = df["sex_male"].astype(int)

# --- fit Cox per cohort -----------------------------------------------------
COVARS = ["prob", "Age", "grade_high", "sex_male"]
LABEL = {"prob": "Model probability", "Age": "Age",
         "grade_high": "grade", "sex_male": "Sex"}
VAR_ORDER = ["prob", "Age", "grade_high", "sex_male"]
COHORTS = ["BTH", "FMUUH", "TCGA"]

rows = []
for coh in COHORTS:
    sub = df[df.cohort == coh][COVARS + ["OS_months", "event"]].copy()
    n, ev = len(sub), int(sub["event"].sum())
    cph = CoxPHFitter().fit(sub, duration_col="OS_months", event_col="event")
    sm = cph.summary
    print(f"\n=== {coh}: n={n} events={ev} ===")
    print(sm[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4).to_string())
    for v in COVARS:
        rows.append(dict(cohort=coh, n=n, events=ev, variable=v,
                         HR=float(sm.loc[v, "exp(coef)"]),
                         lo95=float(sm.loc[v, "exp(coef) lower 95%"]),
                         hi95=float(sm.loc[v, "exp(coef) upper 95%"]),
                         P=float(sm.loc[v, "p"])))
res = pd.DataFrame(rows)
res.to_csv(RES / "multivariate_cox_sex.csv", index=False)
print(f"\nwrote {RES/'multivariate_cox_sex.csv'}")


# --- build booktabs tex (uni_512 format) ------------------------------------
def fmt_hr(r):
    return f"{r.HR:.2f} ({r.lo95:.2f}--{r.hi95:.2f})"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


body = []
for v in VAR_ORDER:
    body.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{LABEL[v]}}}}} \\")
    for coh in COHORTS:
        r = res[(res.variable == v) & (res.cohort == coh)].iloc[0]
        body.append(rf"\quad {coh} & {int(r.n)} & {int(r.events)} & "
                    rf"{fmt_hr(r)} & {fmt_p(r.P)} \\")
    if v != VAR_ORDER[-1]:
        body.append(r"\midrule")
body_tex = "\n".join(body)

tex = r"""\documentclass[a4paper]{article}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{array}
\pagestyle{empty}
\begin{document}

\begin{center}
{\large\bfseries Multivariate Cox analysis}\\[10pt]
\resizebox{0.62\textwidth}{!}{
\begin{tabular}{lcccc}
\toprule
Cohort & n & events & HR (95\% CI) & P \\
\midrule
__BODY__
\bottomrule
\end{tabular}}\\[6pt]
{\small\itshape Model: Hazard $\sim$ Model probability $+$ Age $+$ grade $+$ Sex (slide level; cohort-specific fits). HR for Sex is Male vs.\ Female; grade is WHO grade $\geq$3 vs.\ $<$3.}
\end{center}

\end{document}
"""
tex = tex.replace("__BODY__", body_tex)
out_tex = RES / "multivariate_cox_table.tex"
out_tex.write_text(tex)
print(f"wrote {out_tex}")

# --- compile ----------------------------------------------------------------
for _ in range(2):
    p = subprocess.run(["pdflatex", "-interaction=nonstopmode", out_tex.name],
                       cwd=RES, capture_output=True, text=True)
pdf = RES / "multivariate_cox_table.pdf"
print("PDF:", pdf, "OK" if pdf.exists() else "FAILED")
