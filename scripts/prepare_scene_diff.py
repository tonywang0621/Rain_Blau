import os
import shutil
import cv2
import numpy as np
from pathlib import Path

RAIN_DATA_ROOT = "./dataset/RAIN_data"

def get_sorted_png_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort(key=lambda x: int(Path(x).stem))
    return files

def make_even(img):
    h, w = img.shape[:2]
    return img[:h - h % 2, :w - w % 2]

def process_split(split):
    print(f"\nProcessing {split}...")

    split_dir = Path(RAIN_DATA_ROOT) / split
    input_dir = split_dir / "input"      # rainy
    target_dir = split_dir / "target"    # clean

    if not input_dir.exists():
        print(f"找不到 {input_dir}，跳過")
        return
    if not target_dir.exists():
        print(f"找不到 {target_dir}，跳過")
        return

    files = get_sorted_png_files(input_dir)

    for fname in files:
        stem = Path(fname).stem
        input_path = input_dir / fname
        target_path = target_dir / fname

        if not target_path.exists():
            print(f"skip {fname}: target 不存在")
            continue

        scene_name = f"scene{int(stem):03d}"
        scene_dir = split_dir / scene_name

        blur_dir = scene_dir / "blur"
        sharp_dir = scene_dir / "sharp"
        diff_dir = scene_dir / "ldgp"

        blur_file = blur_dir / "00000.png"
        sharp_file = sharp_dir / "00000.png"
        diff_file = diff_dir / "00000.png"

        os.makedirs(blur_dir, exist_ok=True)
        os.makedirs(sharp_dir, exist_ok=True)
        os.makedirs(diff_dir, exist_ok=True)

        shutil.copy2(input_path, blur_file)
        shutil.copy2(target_path, sharp_file)

        rainy = cv2.imread(str(input_path))
        clean = cv2.imread(str(target_path))

        if rainy is None or clean is None:
            print(f"{scene_name}: 圖片讀不到，跳過")
            continue

        rainy = make_even(rainy)
        clean = make_even(clean)

        if rainy.shape != clean.shape:
            print(f"{scene_name}: rainy / clean 尺寸不同，跳過")
            continue

        diff = cv2.absdiff(rainy, clean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(str(diff_file), diff_gray)

        print(f"{scene_name}: 已建立 blur / sharp / rain-clean差異圖")

for split in ["train", "test"]:
    process_split(split)

print("\nDone.")