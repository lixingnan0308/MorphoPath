"""Feature loading for MorphoPath.

Per-slide features live in ``<data_dir>/<wsi_id>/<*>.h5`` with datasets
``features`` [N, D] and ``coords`` [N, 2]. The label dataframe is built by
``src.dataset.load_label_df`` and must provide columns WSI_ID, ID (patient),
1P19Q (label) and WHO (grade).

Two loading paths:
  - ``preload_data``  : eager — load every slide into RAM (fast training).
  - ``build_lazy_list`` + ``load_item`` : lazy — keep only paths, read on demand.
"""
import os
import h5py
import torch


def preload_data(df, data_dir, max_patches=None):
    """Eagerly load all H5 features into memory.

    If ``max_patches`` is set, deterministically subsample slides exceeding the
    cap (seeded by hash(wsi_id)).
    """
    data_list = []
    print(f"Pre-loading {len(df)} slides into memory...", flush=True)
    for i, (_, row) in enumerate(df.iterrows()):
        wsi_id = str(row["WSI_ID"])
        folder = os.path.join(data_dir, wsi_id)
        h5_files = [f for f in os.listdir(folder) if f.endswith(".h5") and not f.startswith("._")]
        if not h5_files:
            continue
        h5_path = os.path.join(folder, h5_files[0])
        try:
            with h5py.File(h5_path, "r") as f:
                features = torch.from_numpy(f["features"][:])  # [N, D]
                coords = torch.from_numpy(f["coords"][:])      # [N, 2]
        except OSError:
            print(f"  Warning: corrupted H5 file {h5_path}, skipping", flush=True)
            continue
        if max_patches and features.shape[0] > max_patches:
            g = torch.Generator()
            g.manual_seed(abs(hash(wsi_id)) % (2**31))
            idx = torch.randperm(features.shape[0], generator=g)[:max_patches]
            features = features[idx]
            coords = coords[idx]
        data_list.append({
            "features": features,
            "coords": coords,
            "label": int(row["1P19Q"]),
            "wsi_id": wsi_id,
            "patient_id": int(row["ID"]),
            "grade": int(row["WHO"]),
        })
        if (i + 1) % 200 == 0 or i == len(df) - 1:
            print(f"  Loaded {i+1}/{len(df)} slides", flush=True)
    print(f"All {len(data_list)} slides loaded.", flush=True)
    return data_list


def build_lazy_list(df, data_dir, max_patches=None):
    """Build a list of slide records WITHOUT reading features (lazy path)."""
    records = []
    for _, row in df.iterrows():
        wsi_id = str(row["WSI_ID"])
        folder = os.path.join(data_dir, wsi_id)
        h5_files = [f for f in os.listdir(folder) if f.endswith(".h5") and not f.startswith("._")]
        records.append({
            "h5_path": os.path.join(folder, h5_files[0]),
            "label": int(row["1P19Q"]),
            "wsi_id": wsi_id,
            "patient_id": int(row["ID"]),
            "grade": int(row["WHO"]),
            "max_patches": max_patches,
        })
    return records


def load_item(item, max_patches=None):
    """Return a record with ``features``/``coords`` tensors loaded (no-op if eager)."""
    if "features" in item:
        feats, coords = item["features"], item["coords"]
    else:
        with h5py.File(item["h5_path"], "r") as f:
            feats = torch.from_numpy(f["features"][:])
            coords = torch.from_numpy(f["coords"][:])
    cap = max_patches if max_patches is not None else item.get("max_patches")
    if cap and feats.shape[0] > cap:
        g = torch.Generator()
        g.manual_seed(abs(hash(item["wsi_id"])) % (2**31))
        idx = torch.randperm(feats.shape[0], generator=g)[:cap]
        feats = feats[idx]; coords = coords[idx]
    return {**item, "features": feats, "coords": coords}
