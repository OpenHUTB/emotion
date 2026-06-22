#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local image CAM visualizer with official CORnet-S support.

Key features:
  1. Uses the official CORnet-S architecture layout from dicarlolab/CORnet.
  2. Can load official ImageNet pretrained CORnet-S weights from a local .pth file.
  3. Can optionally download the official CORnet-S weights from the official S3 URL.
  4. Supports CAM target layers such as V1.output, V2.output, V4.output, IT.output.

Typical usage after downloading weights:
    python local_image_CAM_cornet_official.py `
      --input "D:/image_CAM/input" `
      --output "D:/image_CAM/cam_outputs_cornet" `
      --model cornet_s `
      --weights "D:/image_CAM/weights/cornet_s-1d3f7974.pth" `
      --target-layer IT.output `
      --method GradCAM `
      --recursive

Optional auto-download usage:
    python local_image_CAM_cornet_official.py `
      --input "D:/image_CAM/input" `
      --output "D:/image_CAM/cam_outputs_cornet" `
      --model cornet_s `
      --official-pretrained `
      --target-layer IT.output `
      --method GradCAM `
      --recursive
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CORNET_S_HASH = "1d3f7974"
CORNET_S_OFFICIAL_URL = "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth"


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CORblock_S(nn.Module):
    scale = 4

    def __init__(self, in_channels: int, out_channels: int, times: int = 1):
        super().__init__()
        self.times = times
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = Identity()

        # Official CORnet-S uses separate BN modules for each recurrent step.
        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(inp)
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)
            x += skip
            x = self.nonlin3(x)

        return self.output(x)


def CORnet_S() -> nn.Module:
    model = nn.Sequential(OrderedDict([
        ("V1", nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm1", nn.BatchNorm2d(64)),
            ("nonlin1", nn.ReLU(inplace=True)),
            ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("conv2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm2", nn.BatchNorm2d(64)),
            ("nonlin2", nn.ReLU(inplace=True)),
            ("output", Identity()),
        ]))),
        ("V2", CORblock_S(64, 128, times=2)),
        ("V4", CORblock_S(128, 256, times=4)),
        ("IT", CORblock_S(256, 512, times=2)),
        ("decoder", nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", Flatten()),
            ("linear", nn.Linear(512, 1000)),
            ("output", Identity()),
        ]))),
    ]))

    # Official initialization pattern.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CAM visualizations for local image files.")
    parser.add_argument("--input", required=True, help="Path to one image file or a folder containing images.")
    parser.add_argument("--output", default="cam_outputs", help="Output folder. Default: cam_outputs")
    parser.add_argument(
        "--model",
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "alexnet", "vgg11", "vgg19", "cornet_s"],
        help="Model architecture. Use cornet_s for official CORnet-S. Default: resnet34",
    )
    parser.add_argument("--weights", default=None, help="Path to a trained checkpoint/state_dict.")
    parser.add_argument(
        "--official-pretrained",
        action="store_true",
        help="For --model cornet_s: download/use official ImageNet pretrained CORnet-S weights.",
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Folder used by --official-pretrained to cache official CORnet-S weights. Default: weights",
    )
    parser.add_argument("--class-names", default=None, help="Optional class-name file, one class name per line.")
    parser.add_argument(
        "--method",
        default="GradCAM",
        choices=["GradCAM", "GradCAM++", "XGradCAM", "ScoreCAM", "AblationCAM", "EigenCAM", "FullGrad"],
        help="CAM method. Default: GradCAM",
    )
    parser.add_argument(
        "--target-layer",
        default=None,
        help="Target layer name. For CORnet-S: V1.output, V2.output, V4.output, IT.output. Default: IT.output.",
    )
    parser.add_argument("--list-layers", action="store_true", help="Print named modules and exit.")
    parser.add_argument("--target-class", type=int, default=None, help="Class index to visualize. If omitted, use Top-1.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions recorded in CSV. Default: 5")
    parser.add_argument("--image-size", type=int, default=224, help="Center-cropped image size. Default: 224")
    parser.add_argument("--resize", type=int, default=256, help="Resize shorter side before crop. Default: 256")
    parser.add_argument("--mean", default="0.485,0.456,0.406", help="Normalization mean. Default: ImageNet mean.")
    parser.add_argument("--std", default="0.229,0.224,0.225", help="Normalization std. Default: ImageNet std.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap transparency. Default: 0.45")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device.")
    parser.add_argument("--no-pretrained", action="store_true", help="For torchvision models only: no ImageNet weights.")
    parser.add_argument("--strict-weights", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively when input is a folder.")
    parser.add_argument("--save-heatmap", action="store_true", help="Also save grayscale CAM heatmap.")
    parser.add_argument("--save-processed-image", action="store_true", help="Also save processed RGB image.")
    return parser.parse_args()


def parse_float_triplet(text: str, name: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"--{name} must contain exactly 3 values, got: {text}")
    return vals


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def maybe_download_official_cornet_s(weights_dir: Union[str, Path]) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst = weights_dir / f"cornet_s-{CORNET_S_HASH}.pth"
    if dst.is_file():
        print(f"[INFO] Official CORnet-S weights already exist: {dst}")
        return dst

    print(f"[INFO] Downloading official CORnet-S weights from: {CORNET_S_OFFICIAL_URL}")
    print(f"[INFO] Saving to: {dst}")
    try:
        torch.hub.download_url_to_file(CORNET_S_OFFICIAL_URL, str(dst), progress=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download official CORnet-S weights automatically. "
            "Download this URL manually in a browser or with PowerShell/curl, then pass it using --weights: "
            f"{CORNET_S_OFFICIAL_URL}\nOriginal error: {exc}"
        )
    return dst


def torch_load_checkpoint(path: Union[str, Path], device: torch.device):
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    try:
        return torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(ckpt_path), map_location=device)


def looks_like_state_dict(obj) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if looks_like_state_dict(ckpt):
        return ckpt
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network", "module"]:
            if key in ckpt:
                value = ckpt[key]
                if isinstance(value, nn.Module):
                    return value.state_dict()
                if looks_like_state_dict(value):
                    return value
    raise ValueError("Could not find a PyTorch state_dict in checkpoint.")


def generate_state_dict_variants(state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    prefixes = ["module.", "model.", "net.", "network.", "backbone.", "_orig_mod."]
    variants = [state_dict]
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            variants.append({(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()})
    # Try repeated all-key stripping.
    cur = state_dict
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cur and all(k.startswith(prefix) for k in cur.keys()):
                cur = {k[len(prefix):]: v for k, v in cur.items()}
                variants.append(cur)
                changed = True
                break
    dedup, seen = [], set()
    for sd in variants:
        sig = tuple(sorted(sd.keys()))
        if sig not in seen:
            seen.add(sig)
            dedup.append(sd)
    return dedup


def choose_best_state_dict(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    model_sd = model.state_dict()

    def score(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        matched = 0
        mismatched = 0
        for k, v in sd.items():
            if k in model_sd:
                if tuple(model_sd[k].shape) == tuple(v.shape):
                    matched += 1
                else:
                    mismatched += 1
        return matched, -mismatched

    variants = generate_state_dict_variants(state_dict)
    return max(variants, key=score)


def load_trained_weights(model: nn.Module, weights_path: Union[str, Path], device: torch.device, strict: bool = False) -> None:
    raw = extract_state_dict(torch_load_checkpoint(weights_path, device))
    best = choose_best_state_dict(raw, model)
    if strict:
        model.load_state_dict(best, strict=True)
        print(f"[INFO] Strictly loaded weights from: {weights_path}")
        return

    model_sd = model.state_dict()
    compatible = OrderedDict()
    unexpected = []
    mismatched = []
    for k, v in best.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        compatible[k] = v
    if not compatible:
        raise RuntimeError("No compatible checkpoint tensors were loaded. Check architecture and checkpoint format.")
    missing, unexpected_after = model.load_state_dict(compatible, strict=False)
    print(f"[INFO] Loaded weights from: {weights_path}")
    print(f"[INFO] Loaded tensors: {len(compatible)} / model tensors: {len(model_sd)}")
    if missing:
        print(f"[WARN] Missing model keys: {len(missing)}; first keys: {list(missing)[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys ignored: {len(unexpected)}; first keys: {unexpected[:10]}")
    if mismatched:
        print(f"[WARN] Shape-mismatched keys ignored: {len(mismatched)}")
        for item in mismatched[:10]:
            print(f"       {item[0]} checkpoint={item[1]} model={item[2]}")
    if unexpected_after:
        print(f"[WARN] Unexpected keys reported by load_state_dict: {len(unexpected_after)}")


def get_weight_enum(model_name: str):
    mapping = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "alexnet": "AlexNet_Weights",
        "vgg11": "VGG11_Weights",
        "vgg19": "VGG19_Weights",
    }
    enum_name = mapping.get(model_name)
    return getattr(models, enum_name, None) if enum_name else None


def load_class_names(path: Optional[str], num_classes: int, default_categories: Optional[Sequence[str]] = None) -> List[str]:
    if path is None:
        if default_categories is not None and len(default_categories) == num_classes:
            return list(default_categories)
        return [f"class_{i}" for i in range(num_classes)]
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Class-name file does not exist: {p}")
    names = []
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                left, right = line.split(",", 1)
                names.append(right.strip() if left.strip().isdigit() else line)
            else:
                names.append(line)
    if len(names) != num_classes:
        fixed = [f"class_{i}" for i in range(num_classes)]
        for i, name in enumerate(names[:num_classes]):
            fixed[i] = name
        print(f"[WARN] Class-name count ({len(names)}) != num_classes ({num_classes}); using fallback names for missing classes.")
        return fixed
    return names


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, List[str], Optional[str]]:
    if args.model == "cornet_s":
        weights_path = args.weights
        if args.official_pretrained:
            weights_path = str(maybe_download_official_cornet_s(args.weights_dir))
        if weights_path is None:
            raise ValueError(
                "For --model cornet_s, use either --official-pretrained or provide --weights <path>. "
                "This prevents accidental CAM generation from random CORnet-S weights."
            )
        model = CORnet_S().eval().to(device)
        load_trained_weights(model, weights_path, device=device, strict=args.strict_weights)
        categories = load_class_names(args.class_names, num_classes=1000)
        return model, categories, weights_path

    constructor = getattr(models, args.model)
    categories = [str(i) for i in range(1000)]
    if args.weights is not None:
        weights_enum = get_weight_enum(args.model)
        model = constructor(weights=None) if weights_enum is not None else constructor(pretrained=False)
        model.eval().to(device)
        load_trained_weights(model, args.weights, device=device, strict=args.strict_weights)
        categories = load_class_names(args.class_names, num_classes=1000)
        return model, categories, args.weights

    pretrained = not args.no_pretrained
    if pretrained:
        weights_enum = get_weight_enum(args.model)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT
            model = constructor(weights=weights)
            categories = list(weights.meta.get("categories", categories))
        else:
            model = constructor(pretrained=True)
    else:
        weights_enum = get_weight_enum(args.model)
        model = constructor(weights=None) if weights_enum is not None else constructor(pretrained=False)
    model.eval().to(device)
    return model, categories, None


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    aliases = {"V1": "V1.output", "V2": "V2.output", "V4": "V4.output", "IT": "IT.output"}
    if name in aliases:
        name = aliases[name]
    if name in modules:
        return modules[name]
    preview = ", ".join([k for k in modules.keys() if k][:80])
    raise ValueError(f"Target layer '{name}' was not found. Use --list-layers. First layers: {preview}")


def select_target_layers(model: nn.Module, model_name: str, target_layer: Optional[str]) -> Tuple[List[nn.Module], str]:
    if target_layer:
        return [get_module_by_name(model, target_layer)], target_layer
    if model_name == "cornet_s":
        return [get_module_by_name(model, "IT.output")], "IT.output"
    if model_name.startswith("resnet"):
        return [model.layer4[-1]], "default_final_conv"
    if model_name == "alexnet":
        return [model.features[-1]], "default_final_conv"
    if model_name.startswith("vgg"):
        return [model.features[-1]], "default_final_conv"
    raise ValueError(f"Unsupported model for target layer selection: {model_name}")


def print_named_layers(model: nn.Module) -> None:
    print("[INFO] Available named modules:")
    for name, module in model.named_modules():
        if name:
            print(f"{name:32s} -> {module.__class__.__name__}")


def build_cam(method_name: str, model: nn.Module, target_layers: Sequence[nn.Module]):
    method_map = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "ScoreCAM": ScoreCAM,
        "AblationCAM": AblationCAM,
        "EigenCAM": EigenCAM,
        "FullGrad": FullGrad,
    }
    return method_map[method_name](model=model, target_layers=list(target_layers))


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    pattern = "**/*" if recursive else "*"
    images = sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return images


def preprocess_pil(image_path: Path, resize: int, image_size: int, device: torch.device, mean: Sequence[float], std: Sequence[float]) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    display_transform = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(image_size)])
    processed = display_transform(image)
    rgb_float = np.float32(np.asarray(processed)) / 255.0
    model_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    return model_transform(image).unsqueeze(0).to(device), rgb_float, processed


def topk_predictions(output: torch.Tensor, categories: Sequence[str], k: int) -> List[Tuple[int, str, float]]:
    probabilities = torch.nn.functional.softmax(output[0], dim=0).detach().cpu()
    k = min(k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=k)
    return [(idx, categories[idx] if idx < len(categories) else str(idx), float(value)) for idx, value in zip(indices.tolist(), values.tolist())]


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)


def save_csv(csv_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = ["image", "cam_method", "model", "weights", "target_layer", "target_class_index", "target_class_name", "target_class_probability", "topk"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_overlay(rgb_float: np.ndarray, grayscale_cam: np.ndarray, alpha: float) -> np.ndarray:
    try:
        return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=alpha)
    except TypeError as exc:
        if "image_weight" not in str(exc):
            raise
        print("[WARN] 当前 pytorch-grad-cam 版本不支持 image_weight，已使用默认叠加强度。")
        return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    mean = parse_float_triplet(args.mean, "mean")
    std = parse_float_triplet(args.std, "std")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model: {args.model}")
    model, categories, loaded_weights_path = load_model(args, device=device)

    if args.list_layers:
        print_named_layers(model)
        return

    target_layers, selected_layer_name = select_target_layers(model, args.model, args.target_layer)
    print(f"[INFO] CAM target layer: {selected_layer_name}")
    cam = build_cam(args.method, model, target_layers)

    image_paths = collect_images(input_path, args.recursive)
    print(f"[INFO] Found {len(image_paths)} image(s).")

    records = []
    for image_path in image_paths:
        print(f"[INFO] Processing: {image_path}")
        input_tensor, rgb_float, processed_pil = preprocess_pil(image_path, args.resize, args.image_size, device, mean, std)
        with torch.no_grad():
            output = model(input_tensor)
        top_rows = topk_predictions(output, categories, args.topk)

        if args.target_class is None:
            target_index, target_name, target_prob = top_rows[0]
        else:
            target_index = int(args.target_class)
            if target_index < 0 or target_index >= output.shape[1]:
                raise ValueError(f"target-class must be in [0, {output.shape[1] - 1}], got {target_index}")
            probabilities = torch.nn.functional.softmax(output[0], dim=0).detach().cpu().numpy()
            target_name = categories[target_index] if target_index < len(categories) else str(target_index)
            target_prob = float(probabilities[target_index])

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_index)])[0, :]
        overlay_rgb = make_overlay(rgb_float, grayscale_cam, alpha=args.alpha)

        method_for_name = args.method.replace("+", "plus")
        layer_for_name = selected_layer_name.replace(".", "_").replace("/", "_")
        stem = safe_stem(image_path)
        overlay_path = output_dir / f"{stem}_{args.model}_{method_for_name}_{layer_for_name}_class{target_index}_overlay.png"
        Image.fromarray(overlay_rgb).save(overlay_path)

        if args.save_heatmap:
            heatmap_path = output_dir / f"{stem}_{args.model}_{method_for_name}_{layer_for_name}_class{target_index}_heatmap.png"
            Image.fromarray(np.uint8(255 * grayscale_cam)).save(heatmap_path)
        if args.save_processed_image:
            processed_pil.save(output_dir / f"{stem}_processed_{args.image_size}.png")

        records.append({
            "image": str(image_path),
            "cam_method": args.method,
            "model": args.model,
            "weights": str(loaded_weights_path) if loaded_weights_path else "",
            "target_layer": selected_layer_name,
            "target_class_index": target_index,
            "target_class_name": target_name,
            "target_class_probability": f"{target_prob:.8f}",
            "topk": "; ".join([f"{idx}:{name}:{prob:.6f}" for idx, name, prob in top_rows]),
        })
        print(f"[OK] Saved overlay: {overlay_path}")
        print(f"     CAM target: {target_index} | {target_name} | prob={target_prob:.4f}")

    csv_method = args.method.replace("+", "plus")
    csv_layer = selected_layer_name.replace(".", "_").replace("/", "_")
    csv_path = output_dir / f"cam_predictions_{args.model}_{csv_method}_{csv_layer}.csv"
    save_csv(csv_path, records)
    print(f"[DONE] Prediction summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
