"""Convert screenshots for the website to WebP.

Usage:
    python tools/optimise_images.py SOURCE [SOURCE ...] --out assets/img/docs

Each PNG/JPEG/WebP is resized to at most --max-width pixels wide (default
1600), saved as <name>.webp in the output folder, and its size is printed
so the width and height can be given to the figure include.
Requires Pillow (pip install pillow).
"""
import argparse
from pathlib import Path

from PIL import Image


def convert(src: Path, out_dir: Path, max_width: int, quality: int) -> tuple[int, int, int]:
    """Save one image as WebP and return (width, height, bytes)."""
    im = Image.open(src)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    dest = out_dir / (src.stem + ".webp")
    im.save(dest, "WEBP", quality=quality, method=6)
    return im.width, im.height, dest.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        w, h, size = convert(src, args.out, args.max_width, args.quality)
        print(f"{src.stem}\t{w}\t{h}\t{size // 1024} kB")


if __name__ == "__main__":
    main()
