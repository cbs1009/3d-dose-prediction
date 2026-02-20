# -*- coding: utf-8 -*-
"""
Inference script for dual-branch MedNeXt dose prediction.

- Input: CT + (X, Y, Rot) condition parsed from case folder name
- Model: MedNeXtDualHead checkpoint trained by train_mednext_muvar.py
- Output: 10-bin predicted dose volumes saved as NIfTI (.nii.gz)

Example
-------
# Bash (Linux/macOS/Git-Bash)
python predict_mednext_muvar.py \
  --input-root "C:/.../Output/Input" \
  --condition-root "C:/.../Output/Output" \
  --checkpoint "C:/.../Training_Results_DualBranch/best_dualbranch.pth" \
  --save-dir "C:/.../Prediction_DualBranch" \
  --compare-with-gt \
  --slice-mode max \
  --viz-window-mode fixed \
  --viz-min 0 \
  --viz-max 65536

# PowerShell (Windows): use backtick (`) for line continuation
python predict_mednext_muvar.py `
  --input-root "C:/.../Output/Input" `
  --condition-root "C:/.../Output/Output" `
  --checkpoint "C:/.../Training_Results_DualBranch/best_dualbranch.pth" `
  --save-dir "C:/.../Prediction_DualBranch" `
  --compare-with-gt `
  --slice-mode max `
  --viz-window-mode fixed `
  --viz-min 0 `
  --viz-max 65536

# Or in PowerShell as a single line
python predict_mednext_muvar.py --input-root "C:/.../Output/Input" --condition-root "C:/.../Output/Output" --checkpoint "C:/.../Training_Results_DualBranch/best_dualbranch.pth" --save-dir "C:/.../Prediction_DualBranch" --compare-with-gt --slice-mode max --viz-window-mode fixed --viz-min 0 --viz-max 65536
"""

import argparse
import glob
import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from train_mednext_muvar import (
    LOG_TRANSFORM,
    MAX_DOSE_VAL,
    MAX_ROT_VAL,
    MAX_SHIFT_VAL,
    TARGET_SIZE,
    MedNeXtDualHead,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-branch MedNeXt inference")
    parser.add_argument("--input-root", type=str, required=True, help="Root containing patient CT folders")
    parser.add_argument(
        "--condition-root",
        type=str,
        required=True,
        help="Root containing patient/case folders (case name includes X/Y/Rot)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--target-size", type=int, nargs=3, default=list(TARGET_SIZE), help="D H W")
    parser.add_argument("--patients", nargs="*", default=None, help="Optional patient IDs to run")
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Also save normalized [0,1] output as .npy (before dose de-normalization)",
    )
    parser.add_argument(
        "--compare-with-gt",
        action="store_true",
        help="Read GT dose from condition root and save middle-slice comparison PNGs + MAE summary.",
    )
    parser.add_argument(
        "--viz-window-mode",
        type=str,
        default="fixed",
        choices=["fixed", "percentile"],
        help="Visualization window mode for pred/GT panel.",
    )
    parser.add_argument(
        "--viz-min",
        type=float,
        default=0.0,
        help="Fixed visualization minimum for pred/GT panel (default: 0).",
    )
    parser.add_argument(
        "--viz-max",
        type=float,
        default=65536.0,
        help="Fixed visualization maximum for pred/GT panel (default: 65536).",
    )
    parser.add_argument(
        "--viz-percentile-low",
        type=float,
        default=1.0,
        help="Low percentile used only when --viz-window-mode percentile.",
    )
    parser.add_argument(
        "--viz-percentile-high",
        type=float,
        default=99.5,
        help="High percentile used only when --viz-window-mode percentile.",
    )
    parser.add_argument(
        "--viz-gamma",
        type=float,
        default=0.8,
        help="Gamma correction for pred/GT visualization (<1 brightens dark regions).",
    )
    parser.add_argument(
        "--slice-mode",
        type=str,
        default="max",
        choices=["mid", "max"],
        help="Comparison slice selection: 'mid' center slice or 'max' highest-activation slice (default: max).",
    )
    return parser.parse_args()


def parse_params(case_str: str) -> Optional[Tuple[float, float, float]]:
    match = re.search(r"X([-\d.]+).*?Y([-\d.]+).*?Rot(?:Z)?([-\d.]+)", case_str)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def to_input_tensor(ct_path: str, params: Tuple[float, float, float], target_size: Tuple[int, int, int]):
    raw_x, raw_y, raw_rot = params

    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    ct_arr = np.clip(ct_arr, -1000, 2000)
    ct_arr = (ct_arr - np.mean(ct_arr)) / (np.std(ct_arr) + 1e-6)

    x_ct = torch.from_numpy(ct_arr).float().unsqueeze(0).unsqueeze(0)
    x_ct = F.interpolate(x_ct, size=target_size, mode="trilinear", align_corners=False).squeeze(0)

    d, h, w = x_ct.shape[1:]
    m_x = torch.full((1, d, h, w), raw_x / MAX_SHIFT_VAL, dtype=torch.float32)
    m_y = torch.full((1, d, h, w), raw_y / MAX_SHIFT_VAL, dtype=torch.float32)
    m_r = torch.full((1, d, h, w), raw_rot / MAX_ROT_VAL, dtype=torch.float32)
    x_input = torch.cat([x_ct, m_x, m_y, m_r], dim=0).unsqueeze(0)

    return x_input, ct_img


def denormalize_dose(pred_norm: np.ndarray) -> np.ndarray:
    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    if LOG_TRANSFORM:
        return np.expm1(pred_norm * np.log1p(MAX_DOSE_VAL)).astype(np.float32)
    return (pred_norm * MAX_DOSE_VAL).astype(np.float32)


def normalize_to_uint8(img: np.ndarray, vmin: float, vmax: float, gamma: float = 1.0) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    arr = (img - vmin) / (vmax - vmin)
    arr = np.clip(arr, 0.0, 1.0)
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    return (arr * 255.0).astype(np.uint8)


def load_gt_dose_stack(case_dir: str) -> Optional[np.ndarray]:
    gt = []
    for i in range(10):
        matches = sorted(glob.glob(os.path.join(case_dir, f"*bin_{i}.dcm")))
        if not matches:
            return None
        arr = sitk.GetArrayFromImage(sitk.ReadImage(matches[0])).astype(np.float32)
        gt.append(np.maximum(arr, 0.0))
    return np.stack(gt, axis=0)


def save_middle_slice_comparison(
    pred_stack: np.ndarray,
    gt_stack: np.ndarray,
    save_path: str,
    viz_window_mode: str,
    viz_min: float,
    viz_max: float,
    pct_low: float,
    pct_high: float,
    gamma: float,
    slice_mode: str = "max",
):
    # Use average over 10 bins and select depth slice
    pred_mean = pred_stack.mean(axis=0)
    gt_mean = gt_stack.mean(axis=0)
    diff = np.abs(pred_mean - gt_mean)

    if slice_mode == "mid":
        z_sel = pred_mean.shape[0] // 2
    else:
        # choose slice with highest GT+Pred energy for easier visual inspection
        profile = (pred_mean + gt_mean).sum(axis=(1, 2))
        z_sel = int(np.argmax(profile))

    pred_sl = pred_mean[z_sel]
    gt_sl = gt_mean[z_sel]
    diff_sl = diff[z_sel]

    if viz_window_mode == "fixed":
        vmin = float(viz_min)
        vmax = float(viz_max)
    else:
        # Robust contrast window from percentiles.
        comb = np.concatenate([pred_sl.ravel(), gt_sl.ravel()])
        vmin = float(np.percentile(comb, pct_low))
        vmax = float(np.percentile(comb, pct_high))
        if vmax <= vmin:
            vmin, vmax = 0.0, float(max(pred_sl.max(), gt_sl.max(), 1e-6))

    pred_u8 = normalize_to_uint8(pred_sl, vmin, vmax, gamma=gamma)
    gt_u8 = normalize_to_uint8(gt_sl, vmin, vmax, gamma=gamma)

    # Diff uses its own window for visibility.
    dmin = float(np.percentile(diff_sl, 1.0))
    dmax = float(np.percentile(diff_sl, 99.5))
    if dmax <= dmin:
        dmin, dmax = 0.0, max(float(diff_sl.max()), 1e-6)
    diff_u8 = normalize_to_uint8(diff_sl, dmin, dmax, gamma=1.0)

    panel = np.concatenate([pred_u8, gt_u8, diff_u8], axis=1)
    sitk.WriteImage(sitk.GetImageFromArray(panel), save_path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    target_size = tuple(args.target_size)

    model = MedNeXtDualHead(in_c=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    all_pids = [d for d in os.listdir(args.condition_root) if os.path.isdir(os.path.join(args.condition_root, d))]
    all_pids = sorted(all_pids)
    if args.patients:
        wanted = set(args.patients)
        all_pids = [p for p in all_pids if p in wanted]

    if not all_pids:
        raise RuntimeError("No patients found to run inference.")

    print(f"[INFO] Device={device} | Patients={len(all_pids)}")
    if args.viz_window_mode == "fixed":
        print(
            f"[INFO] Visualization window: fixed [{args.viz_min:.1f}, {args.viz_max:.1f}], "
            f"gamma={args.viz_gamma:.2f}, slice_mode={args.slice_mode}"
        )
    else:
        print(
            f"[INFO] Visualization window: percentile p{args.viz_percentile_low:.1f}~p{args.viz_percentile_high:.1f}, "
            f"gamma={args.viz_gamma:.2f}, slice_mode={args.slice_mode}"
        )

    total_cases = 0
    total_save_time = 0.0
    total_infer_time = 0.0

    with torch.no_grad():
        for pid in all_pids:
            ct_candidates = glob.glob(os.path.join(args.input_root, pid, "Full_Size", "*Full_CT_Matched.nii*"))
            if not ct_candidates:
                print(f"[WARN] CT not found for patient={pid}, skipped")
                continue
            ct_path = ct_candidates[0]

            case_root = os.path.join(args.condition_root, pid)
            case_dirs = [d for d in os.listdir(case_root) if os.path.isdir(os.path.join(case_root, d))]
            case_dirs = sorted(case_dirs)

            for case_id in case_dirs:
                params = parse_params(case_id)
                if params is None:
                    print(f"[WARN] params parse failed: {pid}/{case_id}, skipped")
                    continue

                x_input, ct_img = to_input_tensor(ct_path, params, target_size)
                x_input = x_input.to(device)

                t_infer = time.perf_counter()
                use_amp = device.type == "cuda"
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred_norm, _, _ = model(x_input)
                else:
                    pred_norm, _, _ = model(x_input)
                infer_sec = time.perf_counter() - t_infer
                total_infer_time += infer_sec

                pred_norm = pred_norm.squeeze(0).cpu().numpy()  # (10, D, H, W)
                pred_dose = denormalize_dose(pred_norm)

                out_case_dir = os.path.join(args.save_dir, pid, case_id)
                os.makedirs(out_case_dir, exist_ok=True)

                # Resize each bin back to original CT shape and save with CT geometry
                orig_size = sitk.GetArrayFromImage(ct_img).shape  # (D, H, W)
                pred_resized = np.empty((pred_dose.shape[0], *orig_size), dtype=np.float32)

                t_save = time.perf_counter()
                for i in range(pred_dose.shape[0]):
                    vol = torch.from_numpy(pred_dose[i]).unsqueeze(0).unsqueeze(0)
                    vol = F.interpolate(vol, size=orig_size, mode="trilinear", align_corners=False)
                    vol_np = vol.squeeze(0).squeeze(0).numpy().astype(np.float32)
                    pred_resized[i] = vol_np

                    out_img = sitk.GetImageFromArray(vol_np)
                    out_img.CopyInformation(ct_img)
                    out_path = os.path.join(out_case_dir, f"pred_bin_{i}.nii.gz")
                    sitk.WriteImage(out_img, out_path)
                save_sec = time.perf_counter() - t_save
                total_save_time += save_sec

                if args.save_normalized:
                    np.save(os.path.join(out_case_dir, "pred_norm_10bin.npy"), pred_norm.astype(np.float32))

                if args.compare_with_gt:
                    gt_case_dir = os.path.join(args.condition_root, pid, case_id)
                    gt_stack = load_gt_dose_stack(gt_case_dir)
                    if gt_stack is None:
                        print(f"[WARN] GT not complete (10 bins): {pid}/{case_id}")
                    else:
                        mae = float(np.mean(np.abs(pred_resized - gt_stack)))
                        comparison_png = os.path.join(out_case_dir, "compare_mid_slice_pred_gt_diff.png")
                        save_middle_slice_comparison(
                            pred_resized,
                            gt_stack,
                            comparison_png,
                            viz_window_mode=args.viz_window_mode,
                            viz_min=args.viz_min,
                            viz_max=args.viz_max,
                            pct_low=args.viz_percentile_low,
                            pct_high=args.viz_percentile_high,
                            gamma=args.viz_gamma,
                            slice_mode=args.slice_mode,
                        )
                        print(
                            f"[CMP] {pid}/{case_id} | MAE={mae:.4f} | comparison panel: {comparison_png} "
                            "(left=pred, mid=gt, right=abs diff; selected slice)"
                        )

                total_cases += 1
                print(
                    f"[OK] saved: {pid}/{case_id} | infer={infer_sec:.2f}s | "
                    f"nii.gz write(10 bins)={save_sec:.2f}s"
                )

    if total_cases > 0:
        print(
            f"[TIME] avg infer/case={total_infer_time / total_cases:.2f}s | "
            f"avg nii.gz write/case={total_save_time / total_cases:.2f}s | cases={total_cases}"
        )

    print("[DONE] Inference finished.")


if __name__ == "__main__":
    main()
