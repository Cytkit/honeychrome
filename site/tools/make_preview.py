"""Build a copy of the website that opens straight from disk.

Usage (from the site folder):
    python tools/make_preview.py

This runs ``bundle exec jekyll build`` into ``_preview/build``, copies the
result to ``_preview/honeychrome-website``, rewrites every site-relative link
("/docs/...") into a relative one ("../../docs/.../index.html") so pages work
when opened with a double-click, and zips the folder to
``_preview/honeychrome-website.zip``. Send the zip to a reviewer: they unzip
it and open ``index.html``.

Use ``--no-build`` to package an existing ``_site`` folder instead.
Requires Ruby and Bundler for the build step (see README.md); the packaging
step needs only Python 3.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
PREVIEW = SITE / "_preview"
ATTR = re.compile(r'(\s(?:href|src)=)(["\'])(/(?!/)[^"\']*)\2')


def relative_link(target: str, page_dir: Path, root: Path) -> str:
    """Turn a site-relative URL into a path relative to the page's folder."""
    path, sep, frag = target.partition("#")
    path, qsep, query = path.partition("?")
    dest = root / path.lstrip("/")
    if path.endswith("/") or dest.is_dir():
        dest = dest / "index.html"
    elif not dest.suffix and dest.with_suffix(".html").exists():
        dest = dest.with_suffix(".html")
    rel = os.path.relpath(dest, page_dir).replace(os.sep, "/")
    return rel + (qsep + query if qsep else "") + (sep + frag if sep else "")


def rewrite_page(page: Path, root: Path) -> None:
    """Rewrite links in one HTML page and record the path back to the root."""
    text = page.read_text(encoding="utf-8")
    page_dir = page.parent
    text = ATTR.sub(lambda m: m.group(1) + m.group(2) + relative_link(m.group(3), page_dir, root) + m.group(2), text)
    to_root = os.path.relpath(root, page_dir).replace(os.sep, "/")
    meta = f'<meta name="hc-root" content="{to_root}/">'
    text = text.replace("<head>", "<head>\n" + meta, 1)
    page.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-build", action="store_true", help="package the existing _site folder")
    args = parser.parse_args()

    if args.no_build:
        built = SITE / "_site"
    else:
        built = PREVIEW / "build"
        env = dict(os.environ, JEKYLL_ENV="development", LANG="C.UTF-8", LC_ALL="C.UTF-8")
        subprocess.run(["bundle", "exec", "jekyll", "build", "--quiet", "-d", str(built)], cwd=SITE, env=env, check=True)
    if not (built / "index.html").exists():
        sys.exit(f"No built site found in {built}")

    out = PREVIEW / "honeychrome-website"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(built, out)
    for page in out.rglob("*.html"):
        rewrite_page(page, out)

    archive = shutil.make_archive(str(PREVIEW / "honeychrome-website"), "zip", root_dir=PREVIEW, base_dir="honeychrome-website")
    print(f"Preview folder: {out}")
    print(f"Zip for reviewers: {archive}")


if __name__ == "__main__":
    main()
