# -*- coding: utf-8 -*-
"""
Dual-branch MedNeXt training script for reducing over-smoothing.

- Input: CT(1) + condition maps(X,Y,Rot) => 4ch
- Output heads:
  * base dose (10ch): coarse / low-frequency
  * detail dose (10ch): high-frequency residual
- Final prediction: clamp(base + detail, 0, 1)

Validation is executed every VAL_INTERVAL epochs (default: 5).
Supports 10-patient split with 9 train / 1 validation.
"""

import gc
import glob
import os
import random
import re
import time
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# 0) Paths & Hyperparams
# ==============================================================================
INPUT_ROOT = r"C:\Users\M292670\Desktop\04_Bo_LU_Jun_Tan\01_MKM_model_prediction\Output\Input"
OUTPUT_ROOT = r"C:\Users\M292670\Desktop\04_Bo_LU_Jun_Tan\01_MKM_model_prediction\Output\Output"
RESULT_SAVE_DIR = r"C:\Users\M292670\Desktop\04_Bo_LU_Jun_Tan\01_MKM_model_prediction\Training_Results_DualBranch"

MAX_DOSE_VAL = 4e10
LOG_TRANSFORM = True

BATCH_SIZE = 1
ACCUMULATION_STEPS = 4
LR = 2e-4
EPOCHS = 300
VAL_INTERVAL = 5

# 9:1 split for 10 patients
VAL_PATIENT_INDEX = 0  # 0~9 사이에서 검증용 환자 인덱스 선택

# loss weights
LAMBDA_BASE = 0.30
LAMBDA_DETAIL = 0.60
LAMBDA_GRAD = 0.10
DETAIL_SCALE = 0.30
DETAIL_THRESH = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (96, 128, 128)

MAX_SHIFT_VAL = 15.0
MAX_ROT_VAL = 5.0

NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
SEED = 42

# Memory safety options (helps prevent intermittent ArrayMemoryError on long runs)
DOSE_TEMP_DTYPE = np.float16  # raw dose temporary arrays in __getitem__
FORCE_GC_EVERY_N_SAMPLES = 64

os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1) Dataset
# ==============================================================================
class MedNeXtDataset(Dataset):
    def __init__(self, input_root: str, output_root: str, patient_ids: List[str], target_size=(96, 128, 128)):
        self.target_size = target_size
        self.samples = []

        for pid in patient_ids:
            ct_dir = os.path.join(input_root, pid, "Full_Size")
            if not os.path.exists(ct_dir):
                continue

            ct_files = glob.glob(os.path.join(ct_dir, "*Full_CT_Matched.nii*"))
            if not ct_files:
                continue
            ct_path = ct_files[0]

            out_patient_dir = os.path.join(output_root, pid)
            if not os.path.exists(out_patient_dir):
                continue

            all_subdirs = [
                d for d in os.listdir(out_patient_dir)
                if os.path.isdir(os.path.join(out_patient_dir, d))
            ]

            for res_dir in all_subdirs:
                params = self._parse_params(res_dir)
                if params is None:
                    continue

                dose_dir_path = os.path.join(out_patient_dir, res_dir)

                # Cache and sort paths once to avoid repeated glob/list allocations in __getitem__
                dose_paths = []
                ok = True
                for i in range(10):
                    matches = sorted(glob.glob(os.path.join(dose_dir_path, f"*bin_{i}.dcm")))
                    if not matches:
                        ok = False
                        break
                    dose_paths.append(matches[0])
                if not ok:
                    continue

                self.samples.append(
                    {
                        "pid": pid,
                        "ct_path": ct_path,
                        "dose_paths": dose_paths,
                        "case_id": res_dir,
                        "params": params,
                    }
                )

        unique_pids = len(set(s["pid"] for s in self.samples))
        print(f"✅ 데이터 로드: {unique_pids}명 유효 | 샘플 수: {len(self.samples)}")

    @staticmethod
    def _parse_params(case_str: str):
        match = re.search(r"X([-\d.]+).*?Y([-\d.]+).*?Rot(?:Z)?([-\d.]+)", case_str)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        raw_x, raw_y, raw_rot = item["params"]

        ct_sitk = sitk.ReadImage(item["ct_path"])
        ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        ct_arr = np.clip(ct_arr, -1000, 2000)
        ct_arr = (ct_arr - np.mean(ct_arr)) / (np.std(ct_arr) + 1e-6)

        x_ct = torch.from_numpy(ct_arr).float().unsqueeze(0).unsqueeze(0)
        x_ct = F.interpolate(x_ct, size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)

        d, h, w = x_ct.shape[1:]
        m_x = torch.full((1, d, h, w), raw_x / MAX_SHIFT_VAL, dtype=torch.float32)
        m_y = torch.full((1, d, h, w), raw_y / MAX_SHIFT_VAL, dtype=torch.float32)
        m_r = torch.full((1, d, h, w), raw_rot / MAX_ROT_VAL, dtype=torch.float32)
        x_input = torch.cat([x_ct, m_x, m_y, m_r], dim=0)

        # Build low-memory target directly in torch tensor (avoid large np.stack temporary)
        target = torch.empty((10, *self.target_size), dtype=torch.float32)
        for i, dose_path in enumerate(item["dose_paths"]):
            d_arr = sitk.GetArrayFromImage(sitk.ReadImage(dose_path)).astype(DOSE_TEMP_DTYPE)
            d_arr = np.maximum(d_arr, 0.0)
            if LOG_TRANSFORM:
                d_arr = np.log1p(d_arr) / np.log1p(MAX_DOSE_VAL)
            else:
                d_arr = d_arr / MAX_DOSE_VAL
            d_arr = np.clip(d_arr, 0.0, 1.0)

            d_tensor = torch.from_numpy(d_arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
            d_tensor = F.interpolate(d_tensor, size=self.target_size, mode="trilinear", align_corners=False)
            target[i] = d_tensor.squeeze(0).squeeze(0)

            del d_arr, d_tensor

        # Occasionally trigger GC to reduce fragmentation on Windows long runs
        if idx % FORCE_GC_EVERY_N_SAMPLES == 0:
            gc.collect()

        return x_input, target, item["pid"], item["case_id"]


# ==============================================================================
# 2) Model
# ==============================================================================
class MedNeXtBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, s: int = 1):
        super().__init__()
        self.res = (s == 1 and in_c == out_c)
        self.dw = nn.Conv3d(in_c, in_c, 7, s, 3, groups=in_c, bias=False)
        self.gn = nn.GroupNorm(1, in_c)
        self.pw1 = nn.Conv3d(in_c, in_c * 4, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(in_c * 4, out_c, 1, bias=False)

    def forward(self, x):
        h = self.pw2(self.act(self.pw1(self.gn(self.dw(x)))))
        return x + h if self.res else h


class MedNeXtDualHead(nn.Module):
    def __init__(self, in_c=4, base=32, depth=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_c, base, 3, padding=1, bias=False),
            nn.GroupNorm(1, base),
        )

        def block(c1, c2, s, n):
            layers = [MedNeXtBlock(c1, c2, s=s)]
            for _ in range(n - 1):
                layers.append(MedNeXtBlock(c2, c2))
            return nn.Sequential(*layers)

        self.e1 = block(base, base, 1, depth)
        self.d1 = nn.Conv3d(base, base * 2, 2, 2)
        self.e2 = block(base * 2, base * 2, 1, depth)
        self.d2 = nn.Conv3d(base * 2, base * 4, 2, 2)
        self.bot = block(base * 4, base * 4, 1, depth)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.dec2 = block(base * 2, base * 2, 1, depth)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, 2)
        self.dec1 = block(base, base, 1, depth)

        self.out_base = nn.Conv3d(base, 10, 1)
        self.out_detail = nn.Conv3d(base, 10, 1)

    def forward(self, x):
        x0 = self.e1(self.stem(x))
        x1 = self.e2(self.d1(x0))
        x2 = self.bot(self.d2(x1))
        y2 = self.dec2(self.u2(x2) + x1)
        y1 = self.dec1(self.u1(y2) + x0)

        base = torch.sigmoid(self.out_base(y1))
        detail = torch.tanh(self.out_detail(y1)) * DETAIL_SCALE
        final = torch.clamp(base + detail, 0.0, 1.0)
        return final, base, detail


# ==============================================================================
# 3) Loss helpers
# ==============================================================================
def make_gaussian_kernel_1d(kernel_size=5, sigma=1.0, device="cpu"):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def gaussian_blur3d(x: torch.Tensor, kernel_size=5, sigma=1.0):
    b, c, d, h, w = x.shape
    g = make_gaussian_kernel_1d(kernel_size, sigma, x.device)

    kx = g.view(1, 1, 1, 1, kernel_size).repeat(c, 1, 1, 1, 1)
    ky = g.view(1, 1, 1, kernel_size, 1).repeat(c, 1, 1, 1, 1)
    kz = g.view(1, 1, kernel_size, 1, 1).repeat(c, 1, 1, 1, 1)

    x = F.conv3d(x, kz, padding=(kernel_size // 2, 0, 0), groups=c)
    x = F.conv3d(x, ky, padding=(0, kernel_size // 2, 0), groups=c)
    x = F.conv3d(x, kx, padding=(0, 0, kernel_size // 2), groups=c)
    return x


def gradient_loss(pred: torch.Tensor, target: torch.Tensor):
    def grad(t):
        dz = torch.abs(t[:, :, 1:, :, :] - t[:, :, :-1, :, :])
        dy = torch.abs(t[:, :, :, 1:, :] - t[:, :, :, :-1, :])
        dx = torch.abs(t[:, :, :, :, 1:] - t[:, :, :, :, :-1])
        return dz, dy, dx

    pdz, pdy, pdx = grad(pred)
    tdz, tdy, tdx = grad(target)
    return (F.l1_loss(pdz, tdz) + F.l1_loss(pdy, tdy) + F.l1_loss(pdx, tdx)) / 3.0


def compute_losses(final, base, detail, y):
    blur_y = gaussian_blur3d(y, kernel_size=5, sigma=1.0)
    detail_target = y - blur_y
    edge_mask = (torch.abs(detail_target) > DETAIL_THRESH).float()

    l_final = F.l1_loss(final, y)
    l_base = F.l1_loss(base, y)

    denom = edge_mask.sum() + 1e-6
    l_detail = (torch.abs(detail - detail_target) * edge_mask).sum() / denom

    l_grad = gradient_loss(final, y)

    total = l_final + (LAMBDA_BASE * l_base) + (LAMBDA_DETAIL * l_detail) + (LAMBDA_GRAD * l_grad)
    return total, l_final, l_base, l_detail, l_grad


# ==============================================================================
# 4) Train / Val
# ==============================================================================
def split_9_train_1_val(all_pids: List[str], val_index: int) -> Tuple[List[str], List[str]]:
    if len(all_pids) < 2:
        raise RuntimeError("Not enough patient folders in INPUT_ROOT.")

    all_pids = sorted(all_pids)
    if val_index < 0 or val_index >= len(all_pids):
        raise ValueError(f"VAL_PATIENT_INDEX must be in [0, {len(all_pids)-1}]")

    val_pid = all_pids[val_index]
    train_pids = [p for p in all_pids if p != val_pid]
    return train_pids, [val_pid]


def main():
    print(f"--- [Start] Dual-branch Training (Device: {DEVICE}) ---")

    all_pids = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    train_pids, val_pids = split_9_train_1_val(all_pids, VAL_PATIENT_INDEX)

    print(f"Train PIDs({len(train_pids)}): {train_pids}")
    print(f"Val PID({len(val_pids)}): {val_pids}")

    train_ds = MedNeXtDataset(INPUT_ROOT, OUTPUT_ROOT, train_pids, TARGET_SIZE)
    val_ds = MedNeXtDataset(INPUT_ROOT, OUTPUT_ROOT, val_pids, TARGET_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = MedNeXtDualHead(in_c=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        st = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        run_total = run_final = run_base = run_detail = run_grad = 0.0

        for i, (x, y, _, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    final, base, detail = model(x)
                    total, l_final, l_base, l_detail, l_grad = compute_losses(final, base, detail, y)
                    loss = total / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                final, base, detail = model(x)
                total, l_final, l_base, l_detail, l_grad = compute_losses(final, base, detail, y)
                loss = total / ACCUMULATION_STEPS
                loss.backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            run_total += float(total.item())
            run_final += float(l_final.item())
            run_base += float(l_base.item())
            run_detail += float(l_detail.item())
            run_grad += float(l_grad.item())

        if (len(train_loader) % ACCUMULATION_STEPS) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        n = max(1, len(train_loader))
        print(
            f"Epoch {epoch:03d} | Train total={run_total/n:.6f} "
            f"final={run_final/n:.6f} base={run_base/n:.6f} detail={run_detail/n:.6f} grad={run_grad/n:.6f} "
            f"| {time.time()-st:.1f}s"
        )

        # every 5 epochs validation result
        if epoch % VAL_INTERVAL == 0:
            model.eval()
            v_total = v_final = v_base = v_detail = v_grad = 0.0
            with torch.no_grad():
                for vx, vy, _, _ in val_loader:
                    vx = vx.to(DEVICE)
                    vy = vy.to(DEVICE)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            final, base, detail = model(vx)
                            total, l_final, l_base, l_detail, l_grad = compute_losses(final, base, detail, vy)
                    else:
                        final, base, detail = model(vx)
                        total, l_final, l_base, l_detail, l_grad = compute_losses(final, base, detail, vy)

                    v_total += float(total.item())
                    v_final += float(l_final.item())
                    v_base += float(l_base.item())
                    v_detail += float(l_detail.item())
                    v_grad += float(l_grad.item())

            m = max(1, len(val_loader))
            avg_v_total = v_total / m
            avg_v_final = v_final / m
            avg_v_base = v_base / m
            avg_v_detail = v_detail / m
            avg_v_grad = v_grad / m

            print(
                f"✅ VAL Epoch {epoch:03d} | total={avg_v_total:.6f} final={avg_v_final:.6f} "
                f"base={avg_v_base:.6f} detail={avg_v_detail:.6f} grad={avg_v_grad:.6f}"
            )

            if avg_v_total < best_val:
                best_val = avg_v_total
                ckpt = os.path.join(RESULT_SAVE_DIR, "best_dualbranch.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"💾 Saved best checkpoint: {ckpt}")

    print("--- [Training Done] ---")


if __name__ == "__main__":
    main()
