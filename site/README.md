# Honeychrome website

The source of https://honeychrome.cytkit.com. It is a Jekyll site, built and
published by GitHub Pages (`.github/workflows/jekyll-gh-pages.yml`) whenever
`main` changes.

## Layout

| Path | What it holds |
|---|---|
| `index.md` | Home page: hero, cards, workflow steps and closing band (front matter) |
| `_home/features/` | Home page feature blocks, one Markdown file each, in file-name order |
| `_home/sections/` | Home page sections after the features, one Markdown file each |
| `download.md`, `science.md`, `releases.md` | Download, Science and Release notes pages |
| `docs/` | Help pages, one Markdown file per page, in a folder per group; `docs/index.md` is the docs home |
| `_data/release.yml` | Current version and download links, used by every download button |
| `_data/docs_nav.yml` | Docs sidebar: groups, page order and titles |
| `_layouts/`, `_includes/` | Page templates, header, banner, footer, figure |
| `assets/css/main.scss`, `assets/js/site.js` | Styles and behaviour (no frameworks) |
| `assets/img/` | Images: `docs/` screenshots, `science/` figures, `brand/` logos |
| `tools/` | Image conversion and offline preview scripts (not published) |
| `src/`, `other/` | Two images kept at their old addresses because the repository README and PyPI page link to them |

## Writing a help page

1. Add `docs/<group>/<page>.md` with front matter:
   ```yaml
   ---
   title: Page title
   lead: One sentence shown under the title.
   ---
   ```
2. Add the page to `_data/docs_nav.yml` in the right group and position.
3. Screenshots: convert them with
   `python tools/optimise_images.py shot.png --out assets/img/docs`
   and insert them with the size the script prints:
   ```liquid
   {% include figure.html src="docs/shot.webp" alt="What the image shows" width="1600" height="1000" caption="Optional caption." %}
   ```
   Two images side by side, at the same height and with one caption, use
   `figure-pair.html` with `src1`, `alt1`, `width1`, `height1`, `src2`,
   `alt2`, `width2`, `height2` and `caption` (see the Statistics section of
   `science.md`).
4. Tips, notes and cautions are blockquotes followed by a class:
   ```markdown
   > Select the controls first to see only their rows.
   {: .tip}
   ```
   Use `{: .note}` or `{: .caution}` for the other two.

## Editing the home page

All the text is in Markdown or in the front matter at the top of a file, so
no HTML needs to be touched.

- **Top of the page** (headline, the line under it, the four cards, the
  workflow steps, the closing band): `index.md`.
- **Feature blocks**: one file per block in `_home/features/`. The front
  matter sets the small label (`eyebrow`), the heading (`title`), the
  screenshot and the link under the bullets; the bullets are the Markdown
  after it:
  ```markdown
  ---
  eyebrow: Gating
  title: Fast plots and gates, with per-sample exceptions
  screenshot: docs/daily_streck_3.webp
  alt: A grid of histograms and density plots
  width: 1600
  height: 1571
  more: Plots and gates
  more_link: /docs/plots-and-gating/plots-and-gates/
  ---
  - Histograms, 2D density and dot plots coloured by gate.
  ```
  Blocks appear in file-name order and alternate sides automatically; rename
  `40-gating.md` to `15-gating.md` to move it up. Without a screenshot, give
  `placeholder: "Screenshot to come: what it will show"` instead.
- **Sections after the features**: `_home/sections/`. A line of three
  asterisks (`***`) splits a section into two columns. The science section
  takes its figures from `stats` in its front matter.
- A link followed by `{: .more}` gets the bold style with an arrow, as in
  `[Programming and plugins](/docs/developers/plugins/){: .more}`.

## The Download page

`download.md` holds the install steps for each platform (front matter) and
the two columns under the platform cards (Markdown, split by `***`). Platform
names, file names and links come from `_data/release.yml`.

## A new release

Edit `_data/release.yml` (version, tag, file names, URLs) and add the notes
to the top of `releases.md`.

## Building locally

Needs Ruby 3 with Bundler (on macOS: `brew install ruby`).

```bash
cd site
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve        # http://localhost:4000, rebuilds on save
```

### A copy to send for review

```bash
python tools/make_preview.py
```

This builds the site and writes `_preview/honeychrome-website.zip`. The
reviewer unzips it and opens `index.html`; every page and the search work
from disk, without a web server.
