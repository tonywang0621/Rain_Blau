# ID-Blau-RainDiffusion

This project is based on the original ID-Blau repository, with modifications for rain streak generation and diffusion-based rain augmentation experiments.

---

# Installation

```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib tqdm tensorboardX pyiqa thop pillow
```

---

# File Structure

```text
ID-Blau-RainDiffusion/
├── diffusion_train.py
├── diffusion_inference.py
├── dataloader.py
├── models/
│   ├── __init__.py
│   ├── diffusion_model.py
│   └── diffusion_network.py
├── utils/
│   └── utils.py
├── scripts/
│   ├── prepare_scene_diff.py
│   └── make_rain_flow.py
├── dataset/
│   ├── RAIN_data/
│   │   ├── train/
│   │   │   ├── input/
│   │   │   └── target/
│   │   └── test/
│   │       ├── input/
│   │       └── target/
│   └── RAIN_flow/
├── experiments/
└── outputs/
```

---

# Dataset Preparation

Prepare dataset as:

```text
dataset/RAIN_data/train/input/1.png
dataset/RAIN_data/train/target/1.png

dataset/RAIN_data/test/input/1.png
dataset/RAIN_data/test/target/1.png
```

Meaning:

```text
input  = rainy image
target = clean image
```

File names must match:

```text
input/1.png  ↔ target/1.png
input/2.png  ↔ target/2.png
```

---

# Step 1: Prepare Scene Format

Run:

```bash
python scripts/prepare_scene_diff.py
```

This converts:

```text
input/1.png
target/1.png
```

into:

```text
dataset/RAIN_data/train/scene001/
├── blur/
│   └── 00000.png
├── sharp/
│   └── 00000.png
└── ldgp/
    └── 00000.png
```

In this project, `ldgp/00000.png` stores:

```text
abs(rainy - clean)
```

---

# Step 2: Generate Condition Map

Run:

```bash
python scripts/make_rain_flow.py
```

This generates:

```text
dataset/RAIN_flow/train/scene001/00000.npy
dataset/RAIN_flow/test/scene001/00000.npy
```

Condition map format:

```text
[0, 0, diff_mask]
```

---

# Step 3: Train

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python diffusion_train.py \
  --data_path ./dataset/RAIN_data \
  --flow_data_path ./dataset/RAIN_flow \
  --dir_path ./experiments/rain_diffusion_v1 \
  --model_name ID_Blau \
  --end_epoch 2000 \
  --batch_size 1 \
  --crop_size 128 \
  --flow_norm False
```

To resume training, use the same `dir_path`:

```bash
CUDA_VISIBLE_DEVICES=0 python diffusion_train.py \
  --data_path ./dataset/RAIN_data \
  --flow_data_path ./dataset/RAIN_flow \
  --dir_path ./experiments/rain_diffusion_v1 \
  --model_name ID_Blau \
  --end_epoch 3000 \
  --batch_size 1 \
  --crop_size 128 \
  --flow_norm False
```

---

# Step 4: Inference

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python diffusion_inference.py \
  --model_path ./experiments/rain_diffusion_v1/last_ID_Blau.pth \
  --dir_path ./outputs/rain_diffusion_v1 \
  --data_path ./dataset/RAIN_data \
  --flow_data_path ./dataset/RAIN_flow \
  --dataset test \
  --sample_timesteps 200 \
  --flow_norm False
```

To run inference on training data:

```bash
--dataset train
```

Recommended sampling settings:

```text
Fast test:    --sample_timesteps 100
Recommended:  --sample_timesteps 200
Stable:       --sample_timesteps 500
```

---

# Output

Inference results will be saved to:

```text
outputs/rain_diffusion_v1/images/
├── sharp/
├── blur/
├── output/
└── flow/
```

Meaning:

```text
sharp  = clean image
blur   = target rainy image
output = generated rainy image
flow   = condition visualization
```
