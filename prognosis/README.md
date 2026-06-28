# prognosis — survival analysis

Survival / prognostic analysis of the MorphoPath 1p/19q prediction across three
cohorts (BTH / FMUUH / TCGA): three-step Kaplan–Meier with log-rank, and
multivariate Cox regression.

Scripts read inputs from `prognosis/data/` and write to `prognosis/results/`
and `prognosis/figures/` (created automatically). `extract_BTH_all_prob.py`
additionally imports the model from the repo's `src/` and uses a trained
checkpoint + the bundled CONCH anchors.

## Pipeline (run in order)

```bash
# 1) MorphoPath forward on BTH slides -> per-slide prediction probabilities
python prognosis/extract_BTH_all_prob.py
#    needs: results/morphopath/best_morphopath_41_seed42.pt, src/conch_{loc,score}.pt,
#           feature .h5 dir (edit FEAT at the top), prognosis/data/BTH_List.xlsx
#    writes: prognosis/data/bth_all_prob.csv

# 2) build canonical cohort tables (BTH + FMUUH + TCGA)
python prognosis/build_cohort.py
#    needs: prognosis/data/{bth_all_prob.csv, FMUUH_List_2.xlsx,
#           source_FMUUH_TCGA_NOBTH.csv, TCGA_List.xlsx}
#    writes: prognosis/data/{cohort_table_all.csv, cohort_table_pos.csv}

# 3) Kaplan–Meier (three-step) + log-rank
python prognosis/km_three_steps.py
#    writes: prognosis/figures/km_*.{pdf,png}, prognosis/results/km_three_steps_logrank.csv

# 4) multivariate Cox (HR ~ model prob + age + grade [+ sex])
python prognosis/cox_multivariate.py          # -> results/multivariate_cox.csv + forest plot
python prognosis/cox_multivariate_table.py    # -> results/multivariate_cox_table.{tex,pdf}  (needs pdflatex)
python prognosis/build_cox_docx.py            # -> results/multivariate_cox_table.docx
```

## Required data (not shipped — provide your own under `prognosis/data/`)

| File | Purpose |
|---|---|
| `BTH_List.xlsx` | BTH cohort with `WSI_ID, ID, WHO, OS, endpoint` (+ `Sex` for the Cox table) |
| `FMUUH_List_2.xlsx` | FMUUH cohort with `final_label` + survival columns |
| `TCGA_List.xlsx` | TCGA cohort list |
| `source_FMUUH_TCGA_NOBTH.csv` | FMUUH/TCGA per-slide predictions + survival |

These contain patient survival data and are intentionally excluded from the
repository.

## Dependencies

`lifelines`, `matplotlib`, `pandas`, `numpy` (KM/Cox); `python-docx` (docx table);
a LaTeX install with `pdflatex` for the booktabs PDF table (optional).
