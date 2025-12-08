#!/usr/bin/env python3
"""
Evaluate multiple pose models (student 8MB, student 2MB, base MobileNet)
using the COCO keypoint protocol. For each model this script reports:
- parameter count
- checkpoint size
- layer-type distribution
- FLOPs (best-effort; optional)
- average inference latency (per image)
- COCO OKS AP plus small/medium/large splits
- mean absolute keypoint error (matched detections)
- peak VRAM usage

Example:
python scripts/evaluate_pose_models.py \
  --labels /path/to/person_keypoints_val2017.json \
  --images-folder /path/to/val2017 \
  --model name=student8,type=student,ckpt=/path/to/student8.pth \
  --model name=student2,type=student2mb,ckpt=/path/to/student2.pth \
  --model name=base,type=mobilenet,ckpt=/path/to/best.pth \
  --output-dir ./eval_runs
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Local repo paths (override via CLI if needed)
DEFAULT_POSE_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lightweight-human-pose-estimation.pytorch")
)

# Add pose repo to import search path before local ai8x imports
import sys

if DEFAULT_POSE_REPO not in sys.path:
    sys.path.insert(0, DEFAULT_POSE_REPO)

import ai8x
from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state


# Student definitions (copied from train_pose_flexible scripts to avoid import side effects)
class StudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 32, 3, stride=2, padding=1, bias=True)
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=True)
        self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=True)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv5 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, bias=True)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv7 = ai8x.FusedConv2dBNReLU(96, 128, 3, stride=1, padding=1, bias=True)
        self.conv8 = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=True)
        self.cpm1 = ai8x.FusedConv2dBNReLU(128, 96, 1, padding=0, bias=True)
        self.cpm2 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
        self.cpm3 = ai8x.FusedConv2dBNReLU(96, 96, 3, padding=1, bias=True)
        self.heat_conv = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.heat_out = ai8x.Conv2d(64, 19, 1, padding=0, bias=True, wide=True)
        self.paf_conv = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.paf_out = ai8x.Conv2d(64, 38, 1, padding=0, bias=True, wide=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.cpm1(x)
        x = self.cpm2(x)
        x = self.cpm3(x)
        h = self.heat_out(self.heat_conv(x))
        p = self.paf_out(self.paf_conv(x))
        return [h, p]


class StudentModel2MB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 24, 3, stride=2, padding=1, bias=True)
        self.conv2 = ai8x.FusedConv2dBNReLU(24, 24, 3, stride=1, padding=1, bias=True)
        self.conv3 = ai8x.FusedConv2dBNReLU(24, 48, 3, stride=1, padding=1, bias=True)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(48, 48, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv5 = ai8x.FusedConv2dBNReLU(48, 64, 3, stride=1, padding=1, bias=True)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=True)
        self.conv7 = ai8x.FusedConv2dBNReLU(64, 96, 3, stride=1, padding=1, bias=True)
        self.conv8 = ai8x.FusedConv2dBNReLU(96, 96, 3, stride=1, padding=1, bias=True)
        self.cpm1 = ai8x.FusedConv2dBNReLU(96, 64, 1, padding=0, bias=True)
        self.cpm2 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=True)
        self.cpm3 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=True)
        self.heat_conv = ai8x.FusedConv2dBNReLU(64, 48, 1, padding=0, bias=True)
        self.heat_out = ai8x.Conv2d(48, 19, 1, padding=0, bias=True, wide=True)
        self.paf_conv = ai8x.FusedConv2dBNReLU(64, 48, 1, padding=0, bias=True)
        self.paf_out = ai8x.Conv2d(48, 38, 1, padding=0, bias=True, wide=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.cpm1(x)
        x = self.cpm2(x)
        x = self.cpm3(x)
        h = self.heat_out(self.heat_conv(x))
        p = self.paf_out(self.paf_conv(x))
        return [h, p]


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue
            cx, cy, score, visibility = 0, 0, 0, 0
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))
    return coco_keypoints, scores


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256,
          device=torch.device("cuda")):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def run_coco_eval(gt_file_path, dt_file_path) -> COCOeval:
    annotation_type = "keypoints"
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()
    return result


def compute_mae_from_eval(coco_eval: COCOeval) -> Optional[float]:
    coco_gt = coco_eval.cocoGt
    coco_dt = coco_eval.cocoDt
    if coco_gt is None or coco_dt is None:
        return None
    total_abs = 0.0
    total_count = 0

    for ev in coco_eval.evalImgs:
        if ev is None:
            continue
        dt_ids = ev["dtIds"]
        if not dt_ids:
            continue
        dt_matches = ev["dtMatches"]
        if dt_matches is None or len(dt_matches) == 0:
            continue
        # Use first OKS threshold matches
        first_thr = dt_matches[0]
        for dt_idx, matched_gt in enumerate(first_thr):
            if matched_gt == 0:
                continue
            gt_id = int(matched_gt)
            dt_id = dt_ids[dt_idx]
            gt = coco_gt.anns[gt_id]["keypoints"]
            dt = coco_dt.anns[dt_id]["keypoints"]
            gt_kpts = np.array(gt, dtype=np.float32).reshape(-1, 3)
            dt_kpts = np.array(dt, dtype=np.float32).reshape(-1, 3)
            visible = gt_kpts[:, 2] > 0
            if not visible.any():
                continue
            diff = np.abs(gt_kpts[visible, :2] - dt_kpts[visible, :2])
            total_abs += diff.sum()
            total_count += diff.size
    if total_count == 0:
        return None
    return float(total_abs / total_count)


def count_layers(model: torch.nn.Module) -> Dict[str, int]:
    counter: Counter = Counter()
    for m in model.modules():
        counter[type(m).__name__] += 1
    return dict(counter)


def estimate_flops(model: torch.nn.Module, device: torch.device, input_size: Tuple[int, int]) -> Optional[float]:
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                model(dummy)
        flops = prof.key_averages().total_average().flops
        return float(flops) if flops is not None else None
    except Exception:
        return None


def load_model(model_type: str):
    if model_type == "mobilenet":
        return PoseEstimationWithMobileNet()
    if model_type == "student":
        return StudentModel()
    if model_type in ("student2mb", "student_2mb"):
        return StudentModel2MB()
    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state", "model"):
            if key in state:
                model.load_state_dict(state[key], strict=False)
                return
        try:
            load_state(model, state)
            return
        except Exception:
            pass
    model.load_state_dict(state, strict=False)


def evaluate_model(
    model_name: str,
    model_type: str,
    checkpoint: str,
    labels: str,
    images_folder: str,
    multiscale: bool,
    device: torch.device,
    base_height: int = 368,
    output_dir: str = ".",
):
    net = load_model(model_type).to(device)
    load_checkpoint(net, checkpoint, device)
    net.eval()

    param_count = sum(p.numel() for p in net.parameters())
    layer_dist = count_layers(net)
    checkpoint_size_mb = os.path.getsize(checkpoint) / (1024 * 1024)

    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scales = [0.5, 1.0, 1.5, 2.0] if multiscale else [1.0]
    stride = 8
    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    per_image_times: List[float] = []

    start = time.time()
    for sample in dataset:
        img = sample["img"]
        file_name = sample["file_name"]
        t0 = time.time()
        avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, stride, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_image_times.append(time.time() - t0)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)
        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
        image_id = int(file_name[0:file_name.rfind(".")])
        for idx in range(len(coco_keypoints)):
            coco_result.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": coco_keypoints[idx],
                    "score": scores[idx],
                }
            )

    total_time = time.time() - start
    output_json = os.path.join(output_dir, f"{model_name}_detections.json")
    with open(output_json, "w") as f:
        json.dump(coco_result, f, indent=2)

    coco_eval = run_coco_eval(labels, output_json)
    mae = compute_mae_from_eval(coco_eval)
    flops = estimate_flops(net, device, (128, 128))
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
    avg_latency_ms = (sum(per_image_times) / len(per_image_times)) * 1000 if per_image_times else None

    ap = coco_eval.stats[0]
    ap_small = coco_eval.stats[3]
    ap_medium = coco_eval.stats[4]
    ap_large = coco_eval.stats[5]
    oks = ap

    return {
        "name": model_name,
        "type": model_type,
        "parameters": int(param_count),
        "checkpoint_mb": round(checkpoint_size_mb, 4),
        "layer_distribution": layer_dist,
        "flops": flops,
        "avg_latency_ms": avg_latency_ms,
        "total_eval_time_s": total_time,
        "oks_ap": oks,
        "ap_small": ap_small,
        "ap_medium": ap_medium,
        "ap_large": ap_large,
        "mae": mae,
        "peak_vram_mb": peak_vram_mb,
        "detections_file": output_json,
    }


def parse_model_arg(arg: str) -> Dict[str, str]:
    """
    Parse model spec in the form name=foo,type=mobilenet,ckpt=/path/to.pth
    """
    parts = arg.split(",")
    parsed = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        parsed[k.strip()] = v.strip()
    required = {"name", "type", "ckpt"}
    if not required.issubset(parsed.keys()):
        missing = required - set(parsed.keys())
        raise ValueError(f"Missing keys in model spec ({arg}): {missing}")
    return {"name": parsed["name"], "type": parsed["type"], "ckpt": parsed["ckpt"]}


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple pose models on COCO keypoints.")
    parser.add_argument("--labels", required=True, help="Path to json with keypoints val labels")
    parser.add_argument("--images-folder", required=True, help="Path to COCO val images folder")
    parser.add_argument("--model", action="append", required=True, help="Model spec: name=foo,type=mobilenet,ckpt=path")
    parser.add_argument("--output-dir", default="./eval_multi", help="Directory to store detections and summary")
    parser.add_argument("--multiscale", action="store_true", help="Average inference over multiple scales")
    parser.add_argument("--pose-repo", default=DEFAULT_POSE_REPO, help="Path to lightweight-human-pose-estimation.pytorch")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation on")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Re-insert pose repo at runtime if user overrides path
    if args.pose_repo not in sys.path:
        sys.path.insert(0, args.pose_repo)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    models = [parse_model_arg(m) for m in args.model]
    summary = []
    for model_cfg in models:
        print(f"\nEvaluating {model_cfg['name']} ({model_cfg['type']}) from {model_cfg['ckpt']}")
        stats = evaluate_model(
            model_cfg["name"],
            model_cfg["type"],
            model_cfg["ckpt"],
            args.labels,
            args.images_folder,
            args.multiscale,
            device,
            output_dir=args.output_dir,
        )
        summary.append(stats)
        print(json.dumps(stats, indent=2))

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
