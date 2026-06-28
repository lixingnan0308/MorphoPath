"""Run a trained MorphoPath checkpoint on the internal test split and report
patient-level metrics (AUC / AUPRC / sensitivity / specificity / PPV / NPV /
F1 / balanced accuracy) at both the Youden threshold and t=0.5.

The split is reproduced from the seed, so it matches training. Example:
  python src/inference.py \
    --ckpt results/morphopath/best_morphopath_41_seed42.pt \
    --conch_loc src/conch_loc.pt --conch_score src/conch_score.pt \
    --n_concepts 6 --n_diagnostic 5 --n_oligo 4 \
    --data_dir path/to/HE_WSI_BTH_512 --label_file path/to/BTH_List.xlsx \
    --seed 42 --youden_json results/morphopath/cv/morphopath_41_oof_youden_seed42.json \
    --tag morphopath_41 --out_dir results/morphopath/internal

The threshold is taken from the CV out-of-fold Youden point
('pooled_oof_youden_threshold', produced by `train.py --mode cv`).
"""
import os, sys, json, argparse
import torch, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_label_df, patient_train_val_test_split
from src.data import preload_data
from src.morphopath import MorphoPath


def load_model(ckpt, conch_loc, conch_score, n_concepts, n_diagnostic, n_oligo, input_dim, device):
    model = MorphoPath(
        tau_init=5.0, n_diagnostic=n_diagnostic, use_normal_anchor=True,
        input_dim=input_dim, proj_dim=512, attn_dim=256,
        n_concepts=n_concepts, n_oligo=n_oligo,
        conch_loc_path=conch_loc, conch_score_path=conch_score,
        dropout=0.1, tau_score=5.0, lambda_loc_bias=0.01,
        sign_constraint=True, attn_mode="residual", tau_loc=1.0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def predict(model, data_list, device):
    probs, labels, pids = [], [], []
    for item in data_list:
        if item["features"].shape[0] <= 5:
            continue
        logit, _, _, _, _ = model(item["features"].to(device), item["coords"].to(device))
        probs.append(torch.sigmoid(logit).item())
        labels.append(item["label"]); pids.append(item["patient_id"])
    agg = pd.DataFrame({"pid": pids, "prob": probs, "label": labels}) \
            .groupby("pid").agg({"prob": "mean", "label": "first"}).reset_index()
    return agg["prob"].values, agg["label"].values.astype(int)


def metrics_at(probs, labels, t):
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "patient_auc": float(roc_auc_score(labels, probs)),
        "patient_auprc": float(average_precision_score(labels, probs)),
        "threshold": float(t),
        "patient_sensitivity": float(tp / max(tp + fn, 1)),
        "patient_specificity": float(tn / max(tn + fp, 1)),
        "patient_ppv": float(tp / max(tp + fp, 1)),
        "patient_npv": float(tn / max(tn + fn, 1)),
        "patient_f1": float(f1_score(labels, preds, zero_division=0)),
        "patient_bal_acc": float(balanced_accuracy_score(labels, preds)),
        "patient_tp": int(tp), "patient_fp": int(fp),
        "patient_tn": int(tn), "patient_fn": int(fn),
        "n_patients": len(labels),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate a MorphoPath checkpoint on the internal test split.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--label_file", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--youden_json", required=True,
                   help="CV out-of-fold youden JSON (*_oof_youden_*.json) with 'pooled_oof_youden_threshold'")
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", default="results/internal")
    p.add_argument("--conch_loc", required=True)
    p.add_argument("--conch_score", required=True)
    p.add_argument("--n_concepts", type=int, default=6)
    p.add_argument("--n_diagnostic", type=int, default=5)
    p.add_argument("--n_oligo", type=int, default=4)
    p.add_argument("--input_dim", type=int, default=1024)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    yj = json.load(open(args.youden_json))
    # threshold = CV out-of-fold Youden point (from train.py --mode cv)
    threshold = yj["pooled_oof_youden_threshold"]

    df = load_label_df(args.label_file)
    df = patient_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=args.seed)
    test_data = preload_data(df[df["split"] == "test"], args.data_dir)

    model = load_model(args.ckpt, args.conch_loc, args.conch_score,
                       args.n_concepts, args.n_diagnostic, args.n_oligo, args.input_dim, device)
    probs, labels = predict(model, test_data, device)

    m_y = metrics_at(probs, labels, threshold)
    m_5 = metrics_at(probs, labels, 0.5)
    out = {"tag": args.tag, "ckpt": args.ckpt, "seed": args.seed,
           "threshold_youden": threshold,
           "internal_test": {"youden": m_y, "t05": m_5, "n_patients": int(m_y["n_patients"])}}
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}_internal.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  AUC={m_y['patient_auc']:.4f} F1(Youden)={m_y['patient_f1']:.4f} F1(0.5)={m_5['patient_f1']:.4f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
