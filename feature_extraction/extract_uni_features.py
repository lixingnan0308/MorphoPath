import os
import torch
import h5py
import logging
import warnings
import numpy as np
import timm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from skimage import color, filters, morphology
from torch.utils.data import Dataset, DataLoader
import random  # 新增：用于随机抽样

warnings.filterwarnings("ignore", message="Color data out of range")

# =========================================================
# 0. OpenSlide 环境配置
# =========================================================
OPENSLIDE_PATH = r'H:\pro\envs\yzy\Library\openslide-bin-4.0.0.5-windows-x64\bin'
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(OPENSLIDE_PATH)
import openslide

# =========================================================
# 1. 动态 Reinhard 归一化类
# =========================================================
class ReinhardNormalizer:
    def __init__(self, target_img_path):
        target_img = Image.open(target_img_path).convert("RGB")
        target_arr = np.array(target_img).astype(np.float32) / 255.0
        target_lab = color.rgb2lab(target_arr)
        self.target_means = np.array([target_lab[:, :, i].mean() for i in range(3)])
        self.target_stds = np.array([target_lab[:, :, i].std() for i in range(3)])

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        lab = color.rgb2lab(img_array)
        for i in range(3):
            mu, sigma = lab[:, :, i].mean(), lab[:, :, i].std()
            lab[:, :, i] = ((lab[:, :, i] - mu) / (sigma + 1e-8)) * self.target_stds[i] + self.target_means[i]
        res = color.lab2rgb(lab) * 255.0
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

# =========================================================
# 2. 组织检测函数
# =========================================================
def get_tissue_coordinates(slide, patch_size_l0, threshold_percent=0.5):
    ds_level = min(2, slide.level_count - 1)
    ds_factor = slide.level_downsamples[ds_level]

    thumb = slide.read_region(
        (0, 0),
        ds_level,
        slide.level_dimensions[ds_level]
    ).convert("RGB")

    thumb_np = np.array(thumb)
    hsv = color.rgb2hsv(thumb_np)
    saturation = hsv[:, :, 1] 
    
    thresh = filters.threshold_otsu(saturation)
    mask = saturation > (thresh * 0.7)

    mask = morphology.remove_small_objects(mask, min_size=100)
    mask = morphology.remove_small_holes(mask, area_threshold=100)

    full_width, full_height = slide.dimensions
    coords = []

    for y in range(0, full_height - patch_size_l0, patch_size_l0):
        for x in range(0, full_width - patch_size_l0, patch_size_l0):
            m_x, m_y = int(x / ds_factor), int(y / ds_factor)
            m_w = int(patch_size_l0 / ds_factor)
            m_h = int(patch_size_l0 / ds_factor)

            if (m_y + m_h > mask.shape[0]) or (m_x + m_w > mask.shape[1]):
                continue

            patch_mask = mask[m_y:m_y + m_h, m_x:m_x + m_w]
            if patch_mask.size == 0: continue

            if np.mean(patch_mask) > threshold_percent:
                coords.append((x, y))
    return coords

# =========================================================
# 3. 数据集加载类
# =========================================================
class WSIPatchDataset(Dataset):
    def __init__(self, slide_path, coords, read_size_l0, transform):
        self.slide_path = slide_path
        self.coords = coords
        self.read_size_l0 = read_size_l0
        self.transform = transform
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)

        x, y = self.coords[idx]
        img = self.slide.read_region(
            (x, y), 0, (self.read_size_l0, self.read_size_l0)
        ).convert("RGB")
        img_resized = img.resize((224, 224), Image.LANCZOS)

        return self.transform(img_resized), torch.tensor([x, y])

# =========================================================
# 4. 主流程
# =========================================================
def process_slide(slide_path, output_base, model, transform, device, num_debug_samples=5):
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_save_dir = os.path.join(output_base, slide_name)
    h5_path = os.path.join(slide_save_dir, f"{slide_name}_features.h5")
    
    # 创建 check 目录
    # debug_dir = os.path.join(slide_save_dir, "check_patches")

    if os.path.exists(h5_path):
        return
    os.makedirs(slide_save_dir, exist_ok=True)
    # os.makedirs(debug_dir, exist_ok=True)

    with openslide.OpenSlide(slide_path) as slide:
        mpp_x = slide.properties.get('openslide.mpp-x')
        if mpp_x is None: 
           res_x = slide.properties.get('tiff.XResolution')
           unit = slide.properties.get('tiff.ResolutionUnit')
           if res_x and unit == 'centimeter':
                # 1 厘米 = 10000 微米，MPP = 10000 / 分辨率
                mpp_x = 10000.0 / float(res_x)
           else:
                mpp_x = 0.5 
                print(f"Warning: {slide_name} has no MPP metadata. Using default 0.5") 

        mpp_x = float(mpp_x)
        read_size_l0 = int(256 * (0.5 / mpp_x))
        coords = get_tissue_coordinates(slide, read_size_l0)

    if not coords: return

    dataset = WSIPatchDataset(slide_path, coords, read_size_l0, transform)
    # 原有的 4 workers 设置
    loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # total_batches = len(loader)
    # save_interval = max(1, total_batches // num_debug_samples)

    all_features, all_coords = [], []
    # saved_count = 0
    
    # # 图像反归一化参数（ImageNet 标准）
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, (batch_imgs, batch_xy) in enumerate(tqdm(loader, desc=f"Processing {slide_name}", leave=False)):
        
        # --- 修改后的确定性保存逻辑 ---
        # 只要当前 batch 索引是间隔的倍数，且还没存够，就保存
        # if i % save_interval == 0 and saved_count < num_debug_samples:
        #     idx = 0  # 取当前 Batch 的第一张图
        #     img_tensor = batch_imgs[idx].cpu()
        #     # 反标准化还原颜色
        #     img_vis = img_tensor * std + mean
        #     img_vis = img_vis.clamp(0, 1).numpy().transpose(1, 2, 0)
        #     img_vis = (img_vis * 255).astype(np.uint8)
            
        #     x, y = batch_xy[idx][0].item(), batch_xy[idx][1].item()
        #     save_path = os.path.join(debug_dir, f"patch_x{x}_y{y}.png")
        #     Image.fromarray(img_vis).save(save_path)
        #     saved_count += 1
        # -------------------------------

        with torch.no_grad():
            batch_imgs = batch_imgs.to(device, dtype=torch.float32)
            features = model(batch_imgs)

        all_features.append(features.cpu().numpy())
        all_coords.append(batch_xy.numpy())

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("features", data=np.concatenate(all_features), compression="gzip")
        f.create_dataset("coords", data=np.concatenate(all_coords), compression="gzip")

# =========================================================
# 5. 执行逻辑
# =========================================================
if __name__ == "__main__":
    import argparse
    _here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Extract UNI (ViT-L/16) patch features from H&E WSIs into per-slide .h5 files.")
    ap.add_argument("--input_wsi_dir", required=True,
                    help="Directory of WSIs (.svs/.ndpi/.tif/.tiff).")
    ap.add_argument("--output_base", required=True,
                    help="Output directory; one <slide>/<slide>_features.h5 per WSI.")
    ap.add_argument("--uni_ckpt", default=os.environ.get("UNI_CKPT_PATH", ""),
                    help="Path to UNI pytorch_model.bin (or set env UNI_CKPT_PATH). "
                         "Download from https://huggingface.co/MahmoodLab/UNI")
    ap.add_argument("--stain_ref", default=os.path.join(_here, "stain_ref.jpeg"),
                    help="Reference image for Reinhard stain normalization.")
    args = ap.parse_args()
    if not args.uni_ckpt:
        raise SystemExit("Provide --uni_ckpt or set env UNI_CKPT_PATH (UNI weights).")

    TARGET_REFERENCE_IMG = args.stain_ref
    UNI_CKPT_PATH = args.uni_ckpt
    INPUT_WSI_DIR = args.input_wsi_dir
    OUTPUT_BASE = args.output_base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = ReinhardNormalizer(TARGET_REFERENCE_IMG)

    uni_transforms = T.Compose([
        normalizer,
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    uni_model = timm.create_model(
        "vit_large_patch16_224", init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    uni_model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu", weights_only=True))
    uni_model.to(device).eval()

    wsi_files = [f for f in os.listdir(INPUT_WSI_DIR) if f.lower().endswith(('.svs', '.ndpi', '.tif', '.tiff'))]

    for f_name in tqdm(wsi_files, desc="Total Progress"):
        slide_path = os.path.join(INPUT_WSI_DIR, f_name)
        try:
            process_slide(slide_path, OUTPUT_BASE, uni_model, uni_transforms, device)
        except Exception as e:
            print(f"Error processing {slide_path}: {e}")
            continue