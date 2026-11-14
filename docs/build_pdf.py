"""Turn `docs/technical-document.md` into the PDF we upload.

    python docs/build_pdf.py --out artifacts/upload/team-nexus.pdf

The markdown is rendered with the figures inlined as data URIs, so the PDF is self-contained and the
SVGs keep their vector quality. Needs headless Chrome, which is the only renderer we found that is
present on a plain developer machine and handles SVG text faithfully. It is not part of the
pipeline's dependency set: the submission file never depends on this script.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CSS = """
@page { size: A4; margin: 16mm 15mm 15mm 15mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 9.3pt; line-height: 1.45;
       color: #1f2933; max-width: none; margin: 0; }
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 2mm 0; color: #0b6e5f; }
h2 { font-size: 12.5pt; margin: 6mm 0 2mm 0; color: #0b6e5f; border-bottom: 1px solid #d6dde5;
     padding-bottom: 1.2mm; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 5mm 0 1.5mm 0; color: #1f2933; page-break-after: avoid; }
p, ul, ol, table, pre { margin: 0 0 2.6mm 0; }
ul, ol { padding-left: 5mm; }
li { margin-bottom: 1mm; }
strong { color: #10232b; }
a { color: #0b6e5f; text-decoration: none; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f1f5f9;
       padding: 0.3mm 1mm; border-radius: 1mm; }
pre { background: #f1f5f9; border: 1px solid #d6dde5; border-radius: 1.5mm; padding: 3mm;
      overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; page-break-inside: avoid; }
th { background: #eef2f7; text-align: left; border-bottom: 1px solid #9aa7b4; }
th, td { padding: 1.4mm 2mm; border-bottom: 1px solid #d6dde5; vertical-align: top; }
blockquote { margin: 0 0 3mm 0; padding: 1mm 0 1mm 4mm; border-left: 2px solid #b9d9d1;
             color: #5f6b7a; }
img { max-width: 100%; display: block; margin: 3mm auto 2mm auto; }
hr { border: none; border-top: 1px solid #d6dde5; margin: 6mm 0; }
"""


def inline_figures(html: str, base: str) -> str:
    """Inline each figure as raw <svg>, so nothing is loaded asynchronously.

    A data-URI <img> renders in a browser screenshot but is still blank when Chrome prints to PDF,
    because the print path does not wait for the image. Inline SVG has no load step, keeps the
    vector text, and prints exactly as it looks on screen.
    """

    def load_svg(src: str) -> str:
        with open(os.path.join(base, src)) as fh:
            svg = fh.read()
        svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
        svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
        # make it scale to the text column while keeping the aspect ratio
        # size it from its own aspect ratio: at most the text column wide, at most 96 mm tall
        size = re.search(r'width="([\d.]+)pt"\s+height="([\d.]+)pt"', svg)
        w_pt, h_pt = (float(size.group(1)), float(size.group(2))) if size else (600.0, 400.0)
        mm = 0.3528
        # wide charts get the full column; a portrait diagram gets more height so its labels stay
        # legible, at the cost of width
        if w_pt / h_pt < 1.3:
            max_w, max_h = 128.0, 142.0
        else:
            max_w, max_h = 150.0, 74.0
        scale = min(max_w / (w_pt * mm), max_h / (h_pt * mm))
        w_mm, h_mm = w_pt * mm * scale, h_pt * mm * scale
        style = (f"width:{w_mm:.0f}mm;height:{h_mm:.0f}mm;display:block;margin:0 auto;")
        svg = re.sub(r'<svg([^>]*)>', lambda m: f'<svg{m.group(1)} style="{style}">', svg, count=1)
        return svg

    def repl(match: re.Match) -> str:
        src = re.search(r'src="([^"]+)"', match.group(1)).group(1)
        if src.endswith(".svg"):
            return "<figure>" + load_svg(src) + "</figure>"
        with open(os.path.join(base, src), "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
        return f'<figure><img src="data:image/png;base64,{payload}" alt="{src}"></figure>'

    return re.sub(r"<p>\s*<img ([^>]*?)\s*/?>\s*</p>", repl, html)


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("no headless Chrome found (tried google-chrome, chromium)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="build the PDF of the technical document")
    parser.add_argument("--source", default=os.path.join(HERE, "technical-document.md"))
    parser.add_argument("--out", default=os.path.join(ROOT, "artifacts", "upload",
                                                      "team-nexus.pdf"))
    args = parser.parse_args(argv)

    with open(args.source) as fh:
        text = fh.read()

    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = inline_figures(html, HERE)
    page = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
           f"<body>{html}</body></html>"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "doc.html")
        pdf_path = os.path.join(tmp, "doc.pdf")
        with open(html_path, "w") as fh:
            fh.write(page)
        subprocess.run(
            [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
             "--no-pdf-header-footer", "--virtual-time-budget=5000",
             f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
            check=True, capture_output=True,
        )
        shutil.copyfile(pdf_path, args.out)

    size = os.path.getsize(args.out)
    with open(args.out, "rb") as fh:
        pages = len(re.findall(rb"/Type\s*/Page[^s]", fh.read()))
    print("wrote {} ({:.0f} KB, {} pages)".format(os.path.relpath(args.out, ROOT), size / 1024, pages))


if __name__ == "__main__":
    main()
