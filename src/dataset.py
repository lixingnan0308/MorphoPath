"""Dataset classes for 1p/19q MIL experiments."""

import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GliomaSlideDataset(Dataset):
    """Dataset for loading pre-extracted UNI features from H5 files."""

    def __init__(self, df, data_dir, label_col="1P19Q", max_patches=None):
        """
        Args:
            df: DataFrame with columns [WSI_ID, ID, 1P19Q, WHO, ...]
            data_dir: Path to HE_WSI_BTH directory
            label_col: Column name for the label
            max_patches: If set, randomly sample this many patches per slide
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_col = label_col
        self.max_patches = max_patches
        # Pre-compute h5 file paths
        self.h5_paths = []
        for _, row in self.df.iterrows():
            wsi_id = str(row["WSI_ID"])
            folder = os.path.join(data_dir, wsi_id)
            h5_files = [f for f in os.listdir(folder) if f.endswith(".h5")]
            self.h5_paths.append(os.path.join(folder, h5_files[0]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_col])
        h5_path = self.h5_paths[idx]

        with h5py.File(h5_path, "r") as f:
            features = torch.from_numpy(f["features"][:])  # [N, 1024]
            coords = torch.from_numpy(f["coords"][:])  # [N, 2]

        if self.max_patches and features.shape[0] > self.max_patches:
            idx_sel = torch.randperm(features.shape[0])[: self.max_patches]
            features = features[idx_sel]
            coords = coords[idx_sel]

        return {
            "features": features,
            "coords": coords,
            "label": label,
            "wsi_id": str(row["WSI_ID"]),
            "patient_id": int(row["ID"]),
            "grade": int(row["WHO"]),
        }


def collate_mil(batch):
    """Custom collate for variable-length bags."""
    return {
        "features": [item["features"] for item in batch],
        "coords": [item["coords"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch]),
        "wsi_id": [item["wsi_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "grade": torch.tensor([item["grade"] for item in batch]),
    }


def load_label_df(excel_path):
    """Load and validate the label DataFrame."""
    df = pd.read_excel(excel_path)
    df["WSI_ID"] = df["WSI_ID"].astype(str)
    assert set(df.columns) >= {"WSI_ID", "ID", "1P19Q", "WHO", "IDH"}
    assert (df["IDH"] == 1).all(), "Expected all IDH-mutant cases"
    return df


def patient_stratified_split(df, n_folds=5, seed=42):
    """
    Patient-stratified K-fold split.
    Ensures all slides from the same patient are in the same fold.
    Stratifies by 1p/19q status.
    """
    from sklearn.model_selection import StratifiedKFold

    # Get patient-level labels (majority vote if multiple slides)
    patient_df = df.groupby("ID").agg(
        {"1P19Q": "first", "WHO": "first"}
    ).reset_index()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patient_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        skf.split(patient_df, patient_df["1P19Q"])
    ):
        patient_df.loc[val_idx, "fold"] = fold

    # Map fold assignments back to slides
    fold_map = dict(zip(patient_df["ID"], patient_df["fold"]))
    df = df.copy()
    df["fold"] = df["ID"].map(fold_map)
    return df


def patient_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Patient-stratified train/val/test split (60/20/20).
    - Train: model training
    - Val: early stopping + hyperparameter selection
    - Test: final evaluation, run ONCE after model is locked
    """
    from sklearn.model_selection import train_test_split

    patient_df = df.groupby("ID").agg(
        {"1P19Q": "first", "WHO": "first"}
    ).reset_index()

    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    trainval_pt, test_pt = train_test_split(
        patient_df, test_size=test_ratio, stratify=patient_df["1P19Q"], random_state=seed
    )
    # Second split: train vs val
    val_frac = val_ratio / (train_ratio + val_ratio)
    train_pt, val_pt = train_test_split(
        trainval_pt, test_size=val_frac, stratify=trainval_pt["1P19Q"], random_state=seed
    )

    # Map back to slides
    df = df.copy()
    train_ids = set(train_pt["ID"])
    val_ids = set(val_pt["ID"])
    test_ids = set(test_pt["ID"])

    def assign_split(patient_id):
        if patient_id in train_ids:
            return "train"
        elif patient_id in val_ids:
            return "val"
        else:
            return "test"

    df["split"] = df["ID"].apply(assign_split)
    return df
