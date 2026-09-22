"""Опорный кэш эмбеддингов DINOv2 для AutoDetect2.

Нужен ровно в одном случае: один и тот же набор кадров используется в нескольких
проектах, и считать по нему признаки заново каждый раз — потерянные часы. Файл
кладётся в `workspace/reference/` (или в папку из `AUTODETECT2_REFERENCE`), и
приложение подхватывает его само; сопоставление идёт по имени файла.

    python tools/build_reference.py images/ --out workspace/reference/site.npz
    python tools/build_reference.py images/ --batch 64 --device cuda

Без GPU считать имеет смысл только на нескольких тысячах кадров: на CPU
трансформер выдаёт единицы изображений в секунду. Если опорного кэша нет,
AutoDetect2 работает без него — просто считает признаки по проекту.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL = "dinov2_vits14_reg"
SIZE = (252, 448)  # (H, W) — кратно 14, как требует патч-сетка DINOv2
EMBED_DIM = 128


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def extract(paths: list[Path], device: str, batch: int) -> np.ndarray:
    import torch
    from PIL import Image

    model = torch.hub.load("facebookresearch/dinov2", MODEL).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def load(path: Path) -> "torch.Tensor":
        img = Image.open(path)
        img.draft("RGB", (SIZE[1], SIZE[0]))
        img = img.convert("RGB").resize((SIZE[1], SIZE[0]))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0)
        return (x - mean) / std

    out: list[np.ndarray] = []
    with torch.no_grad(), ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            tensor = torch.stack(list(pool.map(load, chunk))).to(device)
            res = model.forward_features(tensor)
            feats = torch.cat([res["x_norm_clstoken"], res["x_norm_patchtokens"].mean(1)], dim=1)
            out.append(feats.float().cpu().numpy())
            done = min(start + batch, len(paths))
            print(f"\r  {done}/{len(paths)}", end="", flush=True)

    print()
    return np.concatenate(out).astype(np.float32)


def workspace(feats: np.ndarray, dim: int = EMBED_DIM) -> np.ndarray:
    """L2 -> PCA-whitening -> L2: то же пространство, в котором работает приложение."""
    from sklearn.decomposition import PCA

    dim = int(min(dim, feats.shape[1], max(feats.shape[0] - 1, 2)))
    return l2(PCA(n_components=dim, whiten=True, random_state=0).fit_transform(l2(feats))).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, help="папка с изображениями, обход рекурсивный")
    parser.add_argument("--out", type=Path, default=Path("workspace/reference/reference.npz"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--umap", action="store_true", help="добавить 2D-раскладку для карты данных")
    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Папки нет: {args.images}", file=sys.stderr)
        return 1

    paths = find_images(args.images)
    if not paths:
        print(f"В {args.images} не нашлось изображений", file=sys.stderr)
        return 1
    print(f"Изображений: {len(paths)}")

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    z = workspace(extract(paths, device, args.batch))

    payload = {"names": np.array([p.name for p in paths]), "z": z}
    if args.umap:
        try:
            import umap

            payload["z2d"] = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=0).fit_transform(z).astype(np.float32)
        except ImportError:
            print("umap-learn не установлен — раскладка пропущена", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Готово: {args.out} ({args.out.stat().st_size / 1e6:.1f} МБ, {z.shape[1]}D)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
