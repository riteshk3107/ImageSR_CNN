"""
Benchmark evaluation for Residual SRCNN on common SR datasets.

Fixes common evaluation pitfalls:
- **Border cropping (shave)** before PSNR/SSIM to avoid padding/interpolation artifacts
- **Standard SSIM** via `skimage.metrics.structural_similarity` (Gaussian-weighted + data_range)
- **Metric/visual clarity**: metrics are computed on **Y channel (cropped)**; images are shown in **RGB**
"""

from __future__ import annotations

import argparse
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class ResidualSRCNN(nn.Module):
    """Must match `srcnn.py` ResidualSRCNN for loading weights."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        return self.conv5(out)  # residual


def _list_images(dataset_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        paths.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    return sorted(set(paths))


def _safe_shave_2d(img: np.ndarray, shave: int) -> np.ndarray:
    """Crop `shave` pixels from each border for 2D arrays."""
    if shave <= 0:
        return img
    h, w = img.shape
    if 2 * shave >= h or 2 * shave >= w:
        return img
    return img[shave : h - shave, shave : w - shave]


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def _ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    # Standard SSIM (Gaussian-weighted) comparable to skimage/MATLAB-style evaluation.
    from skimage.metrics import structural_similarity as ssim  # type: ignore

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(
        ssim(
            a,
            b,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


@dataclass
class ImageResult:
    name: str
    path: str
    psnr_y: float
    ssim_y: float
    psnr_y_bicubic: float
    ssim_y_bicubic: float
    shave: int
    original_rgb: Image.Image
    bicubic_rgb: Image.Image
    srcnn_rgb: Image.Image


def _preprocess_hr_to_lr_up_ycbcr(hr_path: str, scale: int) -> Tuple[Image.Image, Image.Image, torch.Tensor, Image.Image, Image.Image]:
    """
    Returns:
      y_hr, y_bicubic_up, inp_tensor (bicubic-up Y in [0,1], 1x1xHxW), cb, cr
    """
    img = Image.open(hr_path).convert("YCbCr")
    y, cb, cr = img.split()
    w, h = y.size
    lr = y.resize((w // scale, h // scale), Image.BICUBIC)
    bicubic_up = lr.resize((w, h), Image.BICUBIC)
    inp = (np.array(bicubic_up).astype(np.float32) / 255.0)
    inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
    return y, bicubic_up, inp_t, cb, cr


def _tensor_to_y_image(t: torch.Tensor) -> Image.Image:
    y = t.squeeze().detach().float().cpu().numpy()
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(y, mode="L")


def _merge_ycbcr_to_rgb(y: Image.Image, cb: Image.Image, cr: Image.Image) -> Image.Image:
    return Image.merge("YCbCr", [y, cb, cr]).convert("RGB")


def evaluate_dataset(
    model: nn.Module,
    device: torch.device,
    dataset_dir: str,
    *,
    scale: int,
    shave: int,
    compute_ssim: bool,
    show_images: bool,
    show_n: Optional[int],
) -> Optional[dict]:
    paths = _list_images(dataset_dir)
    if not paths:
        print(f"Warning: no images found in {dataset_dir}")
        return None

    results: List[ImageResult] = []
    psnrs: List[float] = []
    ssims: List[float] = []
    psnrs_b: List[float] = []
    ssims_b: List[float] = []

    model.eval()
    with torch.no_grad():
        for p in tqdm(paths, desc=f"Evaluating {os.path.basename(dataset_dir)}"):
            y_hr, y_bicubic_up, inp_t, cb, cr = _preprocess_hr_to_lr_up_ycbcr(p, scale)
            inp_t = inp_t.to(device)

            residual = model(inp_t)
            sr = torch.clamp(inp_t + residual, 0.0, 1.0)
            y_sr = _tensor_to_y_image(sr)

            y_hr_np = np.array(y_hr, dtype=np.uint8)
            y_sr_np = np.array(y_sr, dtype=np.uint8)
            y_bi_np = np.array(y_bicubic_up, dtype=np.uint8)

            y_hr_c = _safe_shave_2d(y_hr_np, shave)
            y_sr_c = _safe_shave_2d(y_sr_np, shave)
            y_bi_c = _safe_shave_2d(y_bi_np, shave)

            psnr_sr = _psnr(y_hr_c, y_sr_c)
            psnr_bi = _psnr(y_hr_c, y_bi_c)
            if compute_ssim:
                ssim_sr = _ssim(y_hr_c, y_sr_c)
                ssim_bi = _ssim(y_hr_c, y_bi_c)
            else:
                ssim_sr = float("nan")
                ssim_bi = float("nan")

            psnrs.append(psnr_sr)
            psnrs_b.append(psnr_bi)
            ssims.append(ssim_sr)
            ssims_b.append(ssim_bi)

            name = os.path.basename(p)
            results.append(
                ImageResult(
                    name=name,
                    path=p,
                    psnr_y=psnr_sr,
                    ssim_y=ssim_sr,
                    psnr_y_bicubic=psnr_bi,
                    ssim_y_bicubic=ssim_bi,
                    shave=shave,
                    original_rgb=_merge_ycbcr_to_rgb(y_hr, cb, cr),
                    bicubic_rgb=_merge_ycbcr_to_rgb(y_bicubic_up, cb, cr),
                    srcnn_rgb=_merge_ycbcr_to_rgb(y_sr, cb, cr),
                )
            )

    if show_images:
        _show_results(results, dataset_name=os.path.basename(dataset_dir), show_n=show_n, compute_ssim=compute_ssim)

    return {
        "dataset_dir": dataset_dir,
        "num_images": len(results),
        "shave": shave,
        "avg_psnr_y": float(np.mean(psnrs)) if psnrs else float("nan"),
        "avg_psnr_y_bicubic": float(np.mean(psnrs_b)) if psnrs_b else float("nan"),
        "avg_ssim_y": float(np.mean(ssims)) if ssims else float("nan"),
        "avg_ssim_y_bicubic": float(np.mean(ssims_b)) if ssims_b else float("nan"),
        "image_results": results,
    }


def _show_results(results: List[ImageResult], *, dataset_name: str, show_n: Optional[int], compute_ssim: bool) -> None:
    import matplotlib.pyplot as plt

    if show_n is not None:
        results = results[: max(0, int(show_n))]
    if not results:
        return

    # Grid: one row per image, 3 columns (Original, Bicubic, SRCNN)
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, r in enumerate(results):
        axes[i, 0].imshow(r.original_rgb)
        axes[i, 0].set_title(f"Original (RGB)\n{r.name}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(r.bicubic_rgb)
        bi_title = f"Bicubic (RGB)\nPSNR(Y, shave={r.shave}): {r.psnr_y_bicubic:.2f} dB"
        if compute_ssim:
            bi_title += f" | SSIM(Y): {r.ssim_y_bicubic:.4f}"
        axes[i, 1].set_title(bi_title, fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(r.srcnn_rgb)
        sr_title = f"SRCNN (RGB)\nPSNR(Y, shave={r.shave}): {r.psnr_y:.2f} dB"
        if compute_ssim:
            sr_title += f" | SSIM(Y): {r.ssim_y:.4f}"
        axes[i, 2].set_title(sr_title, fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    plt.suptitle(
        f"{dataset_name}: visuals are RGB; metrics are on Y channel (cropped)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    # Print per-image table (first show_n if set)
    print(f"\n{dataset_name} - Per-image metrics on Y (cropped by shave={results[0].shave})")
    print("-" * 92)
    hdr = f"{'Image':<28} {'PSNR bicubic':>12} {'PSNR srcnn':>12}"
    if compute_ssim:
        hdr += f" {'SSIM bicubic':>12} {'SSIM srcnn':>12}"
    print(hdr)
    print("-" * 92)
    for r in results:
        row = f"{r.name:<28} {r.psnr_y_bicubic:>12.2f} {r.psnr_y:>12.2f}"
        if compute_ssim:
            row += f" {r.ssim_y_bicubic:>12.4f} {r.ssim_y:>12.4f}"
        print(row)
    print("-" * 92)


def evaluate_benchmarks(
    *,
    model_path: str = "srcnn.pth",
    datasets: Optional[Dict[str, str]] = None,
    scale: int = 2,
    shave: Optional[int] = None,
    compute_ssim: bool = True,
    show_images: bool = True,
    show_n: Optional[int] = None,
) -> Dict[str, dict]:
    if datasets is None:
        datasets = {"Set5": "Set5", "Set14": "Set14"}
    if shave is None:
        shave = scale  # IEEE minimum; set to 6*scale if you want that convention.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResidualSRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    out: Dict[str, dict] = {}
    for name, path in datasets.items():
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Warning: {name} directory not found at {path}")
            continue

        print("\n" + "=" * 60)
        print(f"Evaluating {name}")
        print("=" * 60)
        r = evaluate_dataset(
            model,
            device,
            path,
            scale=scale,
            shave=shave,
            compute_ssim=compute_ssim,
            show_images=show_images,
            show_n=show_n,
        )
        if r is None:
            continue

        print(f"\n{name} Summary (metrics on Y, cropped):")
        print(f"- shave: {r['shave']} px")
        print(f"- PSNR bicubic: {r['avg_psnr_y_bicubic']:.2f} dB")
        print(f"- PSNR srcnn  : {r['avg_psnr_y']:.2f} dB")
        if compute_ssim:
            print(f"- SSIM bicubic: {r['avg_ssim_y_bicubic']:.4f}")
            print(f"- SSIM srcnn  : {r['avg_ssim_y']:.4f}")
        out[name] = r
    return out


def evaluate_benchmarks_simple(
    model_path: str = "srcnn.pth",
    set5_path: str = "Set5",
    set14_path: str = "Set14",
    bsd100_path: str = "",
    urban100_path: str = "",
    manga109_path: str = "",
    scale: int = 2,
    shave: Optional[int] = None,
    compute_ssim: bool = True,
    show_images: bool = True,
    show_n: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Simpler API: pass explicit dataset paths instead of a dict.

    Example (in notebook):
        evaluate_benchmarks_simple(
            model_path="srcnn_best.pth",
            set5_path="/kaggle/input/Set5",
            set14_path="/kaggle/input/Set14",
            scale=2,
            shave=12,
        )
    """
    datasets: Dict[str, str] = {
        "Set5": set5_path,
        "Set14": set14_path,
    }
    if bsd100_path:
        datasets["BSD100"] = bsd100_path
    if urban100_path:
        datasets["Urban100"] = urban100_path
    if manga109_path:
        datasets["Manga109"] = manga109_path

    return evaluate_benchmarks(
        model_path=model_path,
        datasets=datasets,
        scale=scale,
        shave=shave,
        compute_ssim=compute_ssim,
        show_images=show_images,
        show_n=show_n,
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Residual SRCNN on benchmark datasets (Y-channel metrics, cropped).")
    p.add_argument("--model", type=str, default="srcnn.pth", help="Path to model weights (.pth)")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor used for degradation (e.g., 2, 3, 4)")
    p.add_argument(
        "--shave",
        type=int,
        default=None,
        help="Pixels to crop from each border before PSNR/SSIM. Default = scale. (Many papers use 6*scale.)",
    )

    p.add_argument("--set5", type=str, default="Set5", help="Path to Set5 directory")
    p.add_argument("--set14", type=str, default="Set14", help="Path to Set14 directory")
    p.add_argument("--bsd100", type=str, default="", help="Path to BSD100 directory (optional)")
    p.add_argument("--urban100", type=str, default="", help="Path to Urban100 directory (optional)")
    p.add_argument("--manga109", type=str, default="", help="Path to Manga109 directory (optional)")

    p.add_argument("--no-ssim", action="store_true", help="Disable SSIM computation")
    p.add_argument("--show-images", action="store_true", default=True, help="Show RGB visual grids")
    p.add_argument("--no-show-images", action="store_false", dest="show_images", help="Do not show images")
    p.add_argument("--show-n", type=int, default=None, help="Only visualize first N images per dataset")
    return p


def main() -> None:
    parser = _build_cli()
    # Ignore unknown args (Jupyter/Colab passes -f ...)
    args, _unknown = parser.parse_known_args()

    datasets: Dict[str, str] = {
        "Set5": args.set5,
        "Set14": args.set14,
    }
    if args.bsd100:
        datasets["BSD100"] = args.bsd100
    if args.urban100:
        datasets["Urban100"] = args.urban100
    if args.manga109:
        datasets["Manga109"] = args.manga109

    evaluate_benchmarks(
        model_path=args.model,
        datasets=datasets,
        scale=args.scale,
        shave=args.shave,
        compute_ssim=not args.no_ssim,
        show_images=args.show_images,
        show_n=args.show_n,
    )


if __name__ == "__main__":
    main()

