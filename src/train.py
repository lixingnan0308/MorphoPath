"""Train MorphoPath (concept-anchored MIL) for 1p/19q codeletion prediction.

Model selection is by validation loss. Supports two modes:
  --mode tvt : train / val / test on a patient-stratified split (final model)
  --mode cv  : 5-fold CV (optionally dev-only) -> out-of-fold Youden threshold

Example:
  python src/train.py \
    --conch_loc src/conch_loc.pt --conch_score src/conch_score.pt \
    --n_concepts 6 --n_diagnostic 5 --n_oligo 4 --concept_config 41 \
    --data_dir path/to/HE_WSI_BTH_512 --label_file path/to/BTH_List.xlsx \
    --lr 5e-5 --wd 1e-5 --lambda_grade 0.0 --seed 42 \
    --mode tvt --epochs 40 --patience 5 --min_epochs 30 \
    --output_dir results/morphopath
"""
import os, sys, json, time, argparse, logging
import numpy as np, torch, torch.nn as nn, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             balanced_accuracy_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_label_df, patient_train_val_test_split
from src.data import preload_data, build_lazy_list, load_item
from src.morphopath import MorphoPath


def setup_logger(output_dir, name="train"):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, f"{name}.log"), mode="w"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    return logger


def morphopath_kwargs(args):
    return dict(
        tau_init=5.0, n_diagnostic=args.n_diagnostic,
        use_normal_anchor=True,
        input_dim=args.input_dim, proj_dim=512, attn_dim=256,
        n_concepts=args.n_concepts, n_oligo=args.n_oligo,
        conch_loc_path=args.conch_loc, conch_score_path=args.conch_score,
        dropout=0.1, tau_score=5.0, lambda_loc_bias=0.01,
        sign_constraint=True, attn_mode="residual", tau_loc=1.0,
    )


def build_model(args, device):
    return MorphoPath(**morphopath_kwargs(args)).to(device)


def build_optimizer(model, args):
    special_ids = {id(model.tau)}
    special_params = [{"params": [model.tau], "lr": args.lr * 10, "weight_decay": 0.0}]
    if hasattr(model, "score_bias"):
        special_ids.add(id(model.score_bias))
        special_params.append({"params": [model.score_bias], "lr": args.lr * 10, "weight_decay": 0.0})
    base_params = [p for p in model.parameters() if id(p) not in special_ids]
    return torch.optim.Adam([
        {"params": base_params, "lr": args.lr, "weight_decay": args.wd},
        *special_params,
    ])


def train_one_epoch(model, train_data, optimizer, criterion, device, args):
    model.train()
    n = len(train_data); indices = torch.randperm(n).tolist()
    losses = {"total": 0, "bce": 0, "sal": 0}
    for idx in indices:
        item = load_item(train_data[idx])
        features = item["features"].to(device); coords = item["coords"].to(device)
        label = torch.tensor([float(item["label"])], device=device)
        grade_label = torch.tensor([float(item["grade"] >= 3)], device=device)
        optimizer.zero_grad()
        logit, logit_grade, cs, attn, _ = model(features, coords)
        l_bce = criterion(logit.unsqueeze(0), label)
        l_grade = criterion(logit_grade.unsqueeze(0), grade_label)
        l_sal = model.loss_saliency()
        l_align = model.loss_align_loc() + model.loss_align_score()
        l_div = model.loss_attn_div(attn)
        loss = l_bce + args.lambda_grade * l_grade + args.lambda_sal * l_sal + 0.1 * l_align + 0.1 * l_div
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses["total"] += loss.item(); losses["bce"] += l_bce.item(); losses["sal"] += l_sal.item()
    return {k: v / n for k, v in losses.items()}


@torch.no_grad()
def eval_model(model, data_list, criterion, device, args):
    model.eval()
    all_probs, all_labels, all_pids = [], [], []
    total_loss = 0; n = 0
    for item in data_list:
        item = load_item(item)
        features = item["features"].to(device); coords = item["coords"].to(device)
        label_t = torch.tensor([float(item["label"])], device=device)
        grade_t = torch.tensor([float(item["grade"] >= 3)], device=device)
        logit, logit_grade, cs, attn, _ = model(features, coords)
        vl = criterion(logit.unsqueeze(0), label_t).item()
        vl += args.lambda_grade * criterion(logit_grade.unsqueeze(0), grade_t).item()
        vl += 0.5 * model.loss_saliency().item()
        total_loss += vl; n += 1
        all_probs.append(torch.sigmoid(logit).cpu().item())
        all_labels.append(item["label"]); all_pids.append(item["patient_id"])
    return _agg_metrics(all_probs, all_labels, all_pids, total_loss / n)


def _agg_metrics(probs, labels, pids, val_loss):
    pdf = pd.DataFrame({"pid": pids, "prob": probs, "label": labels})
    agg = pdf.groupby("pid").agg({"prob": "mean", "label": "first"}).reset_index()
    probs_np = np.array(probs); labels_np = np.array(labels)
    return {
        "val_loss": val_loss,
        "patient_auc": roc_auc_score(agg["label"], agg["prob"]),
        "slide_auc": roc_auc_score(labels_np, probs_np),
        "slide_bal_acc": balanced_accuracy_score(labels_np, (probs_np > 0.5).astype(int)),
    }


@torch.no_grad()
def _predict(model, data_list, device):
    probs, labels, pids = [], [], []
    for item in data_list:
        item = load_item(item)
        logit, _, _, _, _ = model(item["features"].to(device), item["coords"].to(device))
        probs.append(torch.sigmoid(logit).cpu().item())
        labels.append(item["label"]); pids.append(item["patient_id"])
    return probs, labels, pids


def run_tvt(all_data, df, args, device, logger, model_name):
    train_ids = set(df[df["split"] == "train"]["WSI_ID"].astype(str))
    val_ids = set(df[df["split"] == "val"]["WSI_ID"].astype(str))
    test_ids = set(df[df["split"] == "test"]["WSI_ID"].astype(str))
    train_data = [d for d in all_data if d["wsi_id"] in train_ids]
    val_data = [d for d in all_data if d["wsi_id"] in val_ids]
    test_data = [d for d in all_data if d["wsi_id"] in test_ids]
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    logger.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val_loss = float("inf"); best_state = None; best_epoch = 0; patience_ctr = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_dict = train_one_epoch(model, train_data, optimizer, criterion, device, args)
        scheduler.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = eval_model(model, val_data, criterion, device, args)
            improved = metrics["val_loss"] < best_val_loss
            if improved:
                best_val_loss = metrics["val_loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1; patience_ctr = 0
            elif epoch + 1 > args.min_epochs:
                patience_ctr += 1
            marker = " <- BEST" if improved else ""
            logger.info(f"Ep {epoch+1}/{args.epochs} | loss={loss_dict['total']:.4f} "
                        f"bce={loss_dict['bce']:.4f} sal={loss_dict['sal']:.4f} | "
                        f"val_loss={metrics['val_loss']:.4f} AUC={metrics['patient_auc']:.4f}"
                        f"{marker} | {time.time()-t0:.1f}s")
            if epoch + 1 > args.min_epochs and patience_ctr >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}"); break

    logger.info(f"\nBest epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state); model.eval()

    probs, labels, pids = _predict(model, test_data, device)
    probs_np = np.array(probs); labels_np = np.array(labels)
    agg = pd.DataFrame({"pid": pids, "prob": probs, "label": labels}) \
            .groupby("pid").agg({"prob": "mean", "label": "first"}).reset_index()
    test_auc = roc_auc_score(agg["label"], agg["prob"])
    test_auprc = average_precision_score(labels_np, probs_np)
    test_bal = balanced_accuracy_score(labels_np, (probs_np > 0.5).astype(int))
    logger.info(f"*** TEST *** PtAUC={test_auc:.4f} AUPRC={test_auprc:.4f} BalAcc={test_bal:.4f}")

    torch.save(best_state, os.path.join(args.output_dir, f"best_{model_name}_seed{args.seed}.pt"))

    # Youden threshold on validation set
    vp, vl, vpid = _predict(model, val_data, device)
    vagg = pd.DataFrame({"pid": vpid, "prob": vp, "label": vl}) \
             .groupby("pid").agg({"prob": "mean", "label": "first"}).reset_index()
    fpr, tpr, ths = roc_curve(vagg["label"].astype(int), vagg["prob"].values)
    j = tpr - fpr; idx = int(np.argmax(j)); thresh = float(ths[idx])
    logger.info(f"Youden: {thresh:.4f} J={j[idx]:.4f}")

    results = {
        "model": model_name, "selection": "loss", "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "youden_threshold": thresh,
        "test_metrics": {"patient_auc": test_auc, "slide_auprc": test_auprc, "slide_bal_acc": test_bal},
        "args": {k: v for k, v in vars(args).items() if k != "output_dir"},
    }
    json.dump(results, open(os.path.join(args.output_dir, f"{model_name}_tvt_seed{args.seed}.json"), "w"),
              indent=2, default=str)
    json.dump({"threshold": thresh, "youden_index": float(j[idx])},
              open(os.path.join(args.output_dir, f"youden_{model_name}_seed{args.seed}.json"), "w"), indent=2)
    return results


def run_cv(all_data, df, args, device, logger, model_name):
    from sklearn.model_selection import StratifiedKFold
    if args.cv_dev_only:
        df_split = patient_train_val_test_split(df.copy(), train_ratio=0.6, val_ratio=0.2, seed=args.seed)
        dev_df = df_split[df_split["split"] != "test"].copy()
        logger.info(f"CV dev-only: {dev_df['ID'].nunique()} dev patients "
                    f"(excluded {df_split[df_split['split']=='test']['ID'].nunique()} test patients)")
        dev_wsis = set(dev_df["WSI_ID"].astype(str))
        all_data = [d for d in all_data if d["wsi_id"] in dev_wsis]
        patients = dev_df.groupby("ID").agg({"1P19Q": "first"}).reset_index()
    else:
        patients = df.groupby("ID").agg({"1P19Q": "first"}).reset_index()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results = []; oof_records = []
    criterion = nn.BCEWithLogitsLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(patients["ID"], patients["1P19Q"])):
        train_pids = set(patients.iloc[train_idx]["ID"].astype(str))
        val_pids = set(patients.iloc[val_idx]["ID"].astype(str))
        train_wsis = set(df[df["ID"].astype(str).isin(train_pids)]["WSI_ID"].astype(str))
        val_wsis = set(df[df["ID"].astype(str).isin(val_pids)]["WSI_ID"].astype(str))
        train_data = [d for d in all_data if d["wsi_id"] in train_wsis]
        val_data = [d for d in all_data if d["wsi_id"] in val_wsis]
        logger.info(f"\n--- Fold {fold+1}/5 --- Train: {len(train_data)} | Val: {len(val_data)}")

        model = build_model(args, device)
        optimizer = build_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_val_loss = float("inf"); best_state = None; best_epoch = 0; patience_ctr = 0
        for epoch in range(args.epochs):
            train_one_epoch(model, train_data, optimizer, criterion, device, args)
            scheduler.step()
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                metrics = eval_model(model, val_data, criterion, device, args)
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1; patience_ctr = 0
                elif epoch + 1 > args.min_epochs:
                    patience_ctr += 1
                if epoch + 1 > args.min_epochs and patience_ctr >= args.patience:
                    break

        model.load_state_dict(best_state); model.eval()
        final = eval_model(model, val_data, criterion, device, args)
        logger.info(f"  Fold {fold+1} best_ep={best_epoch} AUC={final['patient_auc']:.4f}")
        fold_results.append({"fold": fold + 1, "best_epoch": best_epoch, **final})
        p, l, pid = _predict(model, val_data, device)
        for prob, lab, pi in zip(p, l, pid):
            oof_records.append({"fold": fold + 1, "patient_id": str(pi), "prob": float(prob), "label": int(lab)})

    aucs = [r["patient_auc"] for r in fold_results]
    logger.info(f"\nCV AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    results = {"model": model_name, "cv_folds": fold_results,
               "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
               "args": {k: v for k, v in vars(args).items() if k != "output_dir"}}
    json.dump(results, open(os.path.join(args.output_dir, f"{model_name}_cv_seed{args.seed}.json"), "w"),
              indent=2, default=str)
    if oof_records:
        oof_df = pd.DataFrame(oof_records)
        oof_df.to_csv(os.path.join(args.output_dir, f"{model_name}_oof_preds_seed{args.seed}.csv"), index=False)
        agg = oof_df.groupby("patient_id").agg({"prob": "mean", "label": "first"}).reset_index()
        fpr, tpr, thr = roc_curve(agg["label"].astype(int), agg["prob"])
        j = tpr - fpr; idx = int(j.argmax())
        oof_summary = {"n_oof_patients": int(len(agg)),
                       "pooled_oof_AUC": float(roc_auc_score(agg["label"].astype(int), agg["prob"])),
                       "pooled_oof_youden_threshold": float(thr[idx]),
                       "pooled_oof_youden_J": float(j[idx]),
                       "mean_best_epoch": float(np.mean([r["best_epoch"] for r in fold_results]))}
        logger.info(f"\nPooled OOF: n={oof_summary['n_oof_patients']} "
                    f"AUC={oof_summary['pooled_oof_AUC']:.4f} "
                    f"Youden_thr={oof_summary['pooled_oof_youden_threshold']:.4f}")
        json.dump(oof_summary,
                  open(os.path.join(args.output_dir, f"{model_name}_oof_youden_seed{args.seed}.json"), "w"), indent=2)
    return results


def main():
    p = argparse.ArgumentParser(description="Train MorphoPath for 1p/19q codeletion.")
    p.add_argument("--concept_config", default="41", help="concept dictionary config tag (e.g. 41)")
    p.add_argument("--conch_loc", required=True, help="CONCH morphology-anchor tensor (.pt)")
    p.add_argument("--conch_score", required=True, help="CONCH scoring-anchor tensor (.pt)")
    p.add_argument("--n_concepts", type=int, default=6)
    p.add_argument("--n_diagnostic", type=int, default=5)
    p.add_argument("--n_oligo", type=int, default=4)
    p.add_argument("--data_dir", required=True, help="<data_dir>/<wsi_id>/<*>.h5")
    p.add_argument("--label_file", required=True, help="xlsx with WSI_ID, ID, 1P19Q, WHO")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", default="tvt", choices=["tvt", "cv"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_preload", action="store_true", help="Lazy-load H5 instead of preloading into RAM")
    p.add_argument("--max_patches", type=int, default=None, help="Cap per-slide patch count")
    p.add_argument("--input_dim", type=int, default=1024, help="Feature dim (UNI=1024, CONCH=512, Gigapath=1536)")
    p.add_argument("--lambda_grade", type=float, default=0.0, help="Auxiliary WHO-grade BCE weight (0 = off)")
    p.add_argument("--lambda_sal", type=float, default=0.5, help="Saliency self-distillation weight")
    p.add_argument("--cv_dev_only", action="store_true", help="Restrict CV to the dev split (exclude test)")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = f"morphopath_{args.concept_config}"
    logger = setup_logger(args.output_dir, name=f"{model_name}_{args.mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_name} | Mode: {args.mode} | "
                f"load={'lazy' if args.no_preload else 'preload'}")

    df = load_label_df(args.label_file)
    if args.mode == "tvt":
        df = patient_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=args.seed)
    all_data = (build_lazy_list(df, args.data_dir, max_patches=args.max_patches) if args.no_preload
                else preload_data(df, args.data_dir, max_patches=args.max_patches))

    if args.mode == "tvt":
        run_tvt(all_data, df, args, device, logger, model_name)
    else:
        run_cv(all_data, df, args, device, logger, model_name)
    logger.info("Done!")


if __name__ == "__main__":
    main()
