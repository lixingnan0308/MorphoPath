# MorphoPath: A Robust, Pathology-Prior-Guided AI Framework for Interpretable 1p/19q Co-deletion Prediction in IDH-mutant Glioma

**A diagnostic prior-guided deep-learning framework that incorporates morphology-based pathological knowledge for robust, interpretable prediction of 1p/19q co-deletion from H&E whole-slide images of IDH-mutant gliomas.**

The 1p/19q co-deletion distinguishes oligodendroglioma from astrocytoma in
IDH-mutant diffuse gliomas and drives prognosis and treatment decisions, yet its
determination is costly, slow, and not universally accessible. MorphoPath
addresses this by **embedding pathologists' diagnostic criteria into a pathology
foundation model**: textual descriptions of diagnostic morphology are encoded
into concept anchors, so the model learns clinically meaningful morphological
concepts and links each molecular prediction to interpretable histopathological
evidence.

Across five independent cohorts (1,794 patients, 2,685 whole-slide images),
MorphoPath outperformed conventional multiple-instance learning, reaching an
AUROC of 0.979 on the internal test cohort with strong external generalisation.
It localises the morphological regions behind each prediction in line with
pathologist reasoning, and its scores independently stratify patient survival.

### How it works

MorphoPath is weakly supervised (slide-label only) and keeps its attention
interpretable by **decoupling it into two orthogonal factors**:

1. **Morphology attention** — cosine similarity between each patch and a
   dictionary of **concept anchors** derived from pathologists' diagnostic
   criteria via a pathology foundation model (CONCH) — *"what concept does this
   patch look like?"*, and
2. **A learned saliency gate** — *"is this patch diagnostically relevant?"*

The final patch attention is the (renormalised) product of the two. Concept
scores come from a second anchor dictionary, with a lightweight **residual score
adapter** so the text anchors transfer to other patch encoders (e.g. UNI).

## Repository layout

```
MorphoPath/
├── src/                         # model + training + inference (run from repo root)
│   ├── morphopath_backbone.py   # MorphoPathBackbone — dual-dictionary concept-attention MIL
│   ├── morphopath_attn.py       # MorphoPathAttn(Backbone) — saliency × morphology attention
│   ├── morphopath.py            # MorphoPath(Attn) — + residual score adapter  (the model)
│   ├── dataset.py               # label loading + patient-stratified splits / CV
│   ├── data.py                  # feature .h5 loading (eager preload + lazy)
│   ├── train.py                 # >>> training entry point (tvt / cv, val-loss selection)
│   ├── inference.py             # >>> evaluate a checkpoint on the internal test split
│   ├── visualize.py             # slide-rendering helpers (used by visualization/)
│   ├── conch_loc.pt             # CONCH morphology anchors
│   └── conch_score.pt           # CONCH scoring anchors
├── feature_extraction/          # raw WSI → per-slide UNI feature .h5
│   ├── extract_uni_features.py      # CUDA, H&E
│   └── stain_ref.jpeg               # Reinhard stain-normalisation reference
├── prognosis/                   # survival analysis (BTH / FMUUH / TCGA)
│   ├── extract_BTH_all_prob.py      # MorphoPath forward → per-slide probabilities
│   ├── build_cohort.py              # build canonical cohort tables
│   ├── km_three_steps.py            # three-step Kaplan-Meier + log-rank
│   ├── cox_multivariate.py          # multivariate Cox (HR ~ prob + age + grade)
│   ├── cox_multivariate_table.py    # Cox table → PDF (needs pdflatex)
│   └── build_cox_docx.py            # Cox table → .docx
├── visualization/               # interpretability overlays
│   └── interpret_concept_overlay.py # per-patch concept-share maps on the WSI
├── requirements.txt
└── LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

- **Core** (model/train/inference): PyTorch + the scientific Python stack.
- **Prognosis** needs `lifelines`; **visualisation/feature-extraction** need
  `openslide-python` (plus the OpenSlide C library) and `timm`.
- A GPU (CUDA, or Apple MPS) is recommended; code falls back to CPU.

### Pretrained encoder weights (not in this repo)

Feature extraction uses the **UNI** encoder (`vit_large_patch16_224`, 1024-d)
from [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI). Download
`pytorch_model.bin` and pass it via `--uni_ckpt` (or the `UNI_CKPT_PATH` env var).

## Data format

```
<feature_dir>/<wsi_id>/<wsi_id>_features.h5   # datasets: features [N,1024], coords [N,2]
<label_file>.xlsx                              # columns: WSI_ID, ID (patient), 1P19Q, WHO (grade)
```

Splits are **patient-stratified** (`ID`); metrics are aggregated per patient.
`--input_dim`: UNI = 1024, CONCH = 512, Gigapath = 1536.

## Usage

### 1. (Optional) extract features from raw slides

```bash
python feature_extraction/extract_uni_features.py \
  --input_wsi_dir path/to/slides --output_base path/to/HE_WSI_BTH_512 \
  --uni_ckpt path/to/UNI/pytorch_model.bin
```

### 2. Train

```bash
# 5-fold CV on the dev split -> out-of-fold Youden threshold
python src/train.py --mode cv --cv_dev_only \
  --conch_loc src/conch_loc.pt --conch_score src/conch_score.pt \
  --n_concepts 6 --n_diagnostic 5 --n_oligo 4 --concept_config 41 \
  --data_dir path/to/HE_WSI_BTH_512 --label_file path/to/BTH_List.xlsx \
  --lr 5e-5 --wd 1e-5 --lambda_grade 0.0 --seed 42 \
  --epochs 20 --patience 5 --min_epochs 15 \
  --output_dir results/morphopath/cv

# Train/val/test -> final checkpoint
python src/train.py --mode tvt \
  --conch_loc src/conch_loc.pt --conch_score src/conch_score.pt \
  --n_concepts 6 --n_diagnostic 5 --n_oligo 4 --concept_config 41 \
  --data_dir path/to/HE_WSI_BTH_512 --label_file path/to/BTH_List.xlsx \
  --lr 5e-5 --wd 1e-5 --lambda_grade 0.0 --seed 42 \
  --epochs 40 --patience 5 --min_epochs 30 \
  --output_dir results/morphopath
```

### 3. Inference (internal test metrics)

```bash
python src/inference.py \
  --ckpt results/morphopath/best_morphopath_41_seed42.pt \
  --conch_loc src/conch_loc.pt --conch_score src/conch_score.pt \
  --n_concepts 6 --n_diagnostic 5 --n_oligo 4 \
  --data_dir path/to/HE_WSI_BTH_512 --label_file path/to/BTH_List.xlsx \
  --seed 42 --youden_json results/morphopath/cv/morphopath_41_oof_youden_seed42.json \
  --tag morphopath_41 --out_dir results/morphopath/internal
```

### 4. Interpretability visualisation

```bash
python visualization/interpret_concept_overlay.py --help   # --label TP | TN, --wsi_id ...
```

### 5. Prognosis (survival analysis)

```bash
python prognosis/extract_BTH_all_prob.py   # MorphoPath forward -> per-slide probabilities
python prognosis/build_cohort.py           # cohort tables (BTH / FMUUH / TCGA)
python prognosis/km_three_steps.py         # three-step Kaplan-Meier + log-rank
python prognosis/cox_multivariate.py       # multivariate Cox + forest plot
```

Provide your cohort/survival files under `prognosis/data/` first — see
`prognosis/README.md` for the required columns and the full pipeline order.

## Notes

- **Best epoch** is selected by validation loss (not AUC).
- `--lambda_grade 0.0` is the no-grade setting (auxiliary WHO-grade head off).
- Concept anchors are bundled (`src/conch_loc.pt`, `src/conch_score.pt`); the
  `41` config = 6 concepts (5 diagnostic + normal anchor), 4 oligo-leaning.

## Citation

If you use this code, please cite the MorphoPath paper. *(citation / BibTeX to be added)*
