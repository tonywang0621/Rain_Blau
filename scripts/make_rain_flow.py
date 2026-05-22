import os
import cv2
import numpy as np
from pathlib import Path

RAIN_DATA_ROOT = "./dataset/RAIN_data"
RAIN_FLOW_ROOT = "./dataset/RAIN_flow"

def make_even(img):
    h, w = img.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    return img[:new_h, :new_w]

def process_split(split):
    split_dir = Path(RAIN_DATA_ROOT) / split
    flow_split_dir = Path(RAIN_FLOW_ROOT) / split

    if not split_dir.exists():
        print(f"[{split}] 不存在，跳過")
        return

    scene_dirs = sorted([
        d for d in split_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene")
    ])

    print(f"\nProcessing {split}...")
    print(f"找到 {len(scene_dirs)} 個 scene")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        clean_path = scene_dir / "sharp" / "00000.png"
        rainy_path = scene_dir / "blur" / "00000.png"
        ldgp_mask_path = scene_dir / "ldgp" / "00000.png"
        save_npy_path = flow_split_dir / scene_name / "00000.npy"

        if save_npy_path.exists():
            print(f"{scene_name}: npy 已存在，跳過")
            continue

        missing = []
        if not clean_path.exists():
            missing.append("sharp/00000.png")
        if not rainy_path.exists():
            missing.append("blur/00000.png")
        if not ldgp_mask_path.exists():
            missing.append("ldgp/00000.png")

        if missing:
            print(f"{scene_name}: 缺少 {missing}，跳過")
            continue

        os.makedirs(save_npy_path.parent, exist_ok=True)

        clean = cv2.imread(str(clean_path))
        rainy = cv2.imread(str(rainy_path))
        mask = cv2.imread(str(ldgp_mask_path), cv2.IMREAD_GRAYSCALE)

        if clean is None:
            print(f"{scene_name}: clean 讀不到，跳過")
            continue
        if rainy is None:
            print(f"{scene_name}: rainy 讀不到，跳過")
            continue
        if mask is None:
            print(f"{scene_name}: ldgp 讀不到，跳過")
            continue

        if clean.shape != rainy.shape:
            print(f"{scene_name}: clean / rainy 尺寸不同，跳過")
            continue

        clean = make_even(clean)
        rainy = make_even(rainy)
        mask = make_even(mask)

        H_img, W_img = clean.shape[:2]

        # LDGP mask normalize 到 [0,1]
        mask = mask.astype(np.float32) / 255.0
        mask = np.sqrt(mask)
        mask = np.clip(mask, 0, 1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

        zero = np.zeros_like(mask, dtype=np.float32)

        # condition = [mask, 0, 0]
        rain_cond = np.stack([
            zero,
            zero,
            mask
        ], axis=0).astype(np.float32)

        rain_cond = np.nan_to_num(
            rain_cond,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        print(
            scene_name,
            "rain_cond range:",
            rain_cond.min(),
            rain_cond.max(),
            rain_cond.mean(),
            "has_nan:",
            np.isnan(rain_cond).any()
        )

        np.save(str(save_npy_path), rain_cond)
        print(f"{scene_name}: 已生成 {save_npy_path}")

for split in ["train", "test"]:
    process_split(split)

print("\nDone.")