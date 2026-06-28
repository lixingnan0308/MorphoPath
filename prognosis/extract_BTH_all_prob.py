"""Extract BTH inference prob for ALL slides (1p/19q + AND -).

Mirror of extract_contrib_BTH.py but without the 1P19Q==1 filter.
Uses the trained MorphoPath (no-grade) checkpoint.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, h5py
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]  # repo root
PROG = Path(__file__).resolve().parent     # prognosis/
sys.path.insert(0, str(REPO))
from src.morphopath import MorphoPath

CKPT  = REPO / "results/morphopath/best_morphopath_41_seed42.pt"
LOC   = REPO / "src/conch_loc.pt"
SCORE = REPO / "src/conch_score.pt"
FEAT  = REPO / "data/HE_WSI_BTH_512"   # set to your per-slide feature .h5 dir
LIST  = PROG / "data/BTH_List.xlsx"
OUT   = PROG / "data/bth_all_prob.csv"

N_OLIGO = 4; N_DIAG = 5

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] device={device}  ckpt={CKPT.name}")

m = MorphoPath(tau_init=5.0, n_diagnostic=N_DIAG, use_normal_anchor=True,
    input_dim=1024, proj_dim=512, attn_dim=256, n_concepts=6, n_oligo=N_OLIGO,
    conch_loc_path=str(LOC), conch_score_path=str(SCORE),
    dropout=0.1, tau_score=5.0, lambda_loc_bias=0.01,
    sign_constraint=True, attn_mode="residual", tau_loc=1.0).to(device)
m.load_state_dict(torch.load(str(CKPT), map_location=device, weights_only=True))
m.eval()

df = pd.read_excel(LIST)
df["wsi_id"] = df["WSI_ID"].astype(str)
df["patient_id"] = df["ID"].astype(str)
df["OS_days"] = pd.to_numeric(df["OS"], errors="coerce")
df["event"] = pd.to_numeric(df["endpoint"], errors="coerce")
df["WHO_grade"] = pd.to_numeric(df["WHO"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["label_1p19q"] = pd.to_numeric(df["1P19Q"], errors="coerce")

keep = df["OS_days"].notna() & df["event"].notna() & df["Age"].notna() & \
       df["WHO_grade"].notna() & df["label_1p19q"].notna()
df_use = df[keep].copy().reset_index(drop=True)
print(f"[data] BTH with OS+age+grade+label: {len(df_use)} slides "
      f"({(df_use['label_1p19q']==1).sum()} pos / {(df_use['label_1p19q']==0).sum()} neg)")

rows = []
for i, r in df_use.iterrows():
    w = str(r["wsi_id"]); folder = FEAT/w
    if not folder.is_dir():
        print(f"[skip] {w}: no feature folder")
        continue
    h5s = [p for p in folder.glob("*_features.h5") if not p.name.startswith("._")]
    if not h5s:
        print(f"[skip] {w}: no h5")
        continue
    try:
        with h5py.File(h5s[0], "r") as f:
            feat = torch.from_numpy(f["features"][:]).float().to(device)
            co = torch.from_numpy(f["coords"][:]).float().to(device)
    except OSError as e:
        print(f"[skip] {w}: {e}")
        continue
    if feat.shape[0] <= 5:
        continue

    with torch.no_grad():
        logit, _, cs, attn, _ = m(feat, co)
        prob = float(torch.sigmoid(logit).item())

    rows.append({
        "cohort": "BTH",
        "patient_id": str(r["patient_id"]),
        "wsi_id": w,
        "label_1p19q": int(r["label_1p19q"]),
        "OS_days": float(r["OS_days"]),
        "event": int(r["event"]),
        "Age": float(r["Age"]),
        "WHO_grade": float(r["WHO_grade"]),
        "prob": round(prob, 4),
    })
    if (len(rows)) % 50 == 0:
        print(f"  [{len(rows)}/{len(df_use)}] last wsi={w} prob={prob:.3f}")

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: {len(out_df)} rows")
if len(out_df) > 0:
    print(out_df.groupby("label_1p19q").agg(
        slides=("wsi_id", "size"),
        patients=("patient_id", "nunique"),
        events=("event", "sum")).to_string())
