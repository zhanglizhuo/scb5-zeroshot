"""Build a Word version of the SCB5 zero-shot paper.

Strategy:
1. Rasterize PDF figures to PNG so Word can render them.
2. Preprocess the MDPI .tex into a pandoc-friendly variant (standard \title/\author/abstract).
3. Run pypandoc to produce .docx.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pypandoc

PAPER_DIR = Path(__file__).resolve().parent
SRC_TEX = PAPER_DIR / "scb5_zeroshot_paper.tex"
FIG_DIR = PAPER_DIR / "figures"
PNG_DIR = PAPER_DIR / "figures_png"
TMP_TEX = PAPER_DIR / "_scb5_zeroshot_paper_pandoc.tex"
OUT_DOCX = PAPER_DIR / "scb5_zeroshot_paper.docx"
TIKZ_DIR = PNG_DIR / "tikz_export"


def rasterize_pdfs() -> None:
    PNG_DIR.mkdir(exist_ok=True)
    for pdf in sorted(FIG_DIR.glob("*.pdf")):
        png = PNG_DIR / (pdf.stem + ".png")
        if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
            continue
        # pdftocairo writes <out>.png for single-page PDFs when -singlefile is used
        subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-r",
                "200",
                "-singlefile",
                str(pdf),
                str(PNG_DIR / pdf.stem),
            ],
            check=True,
        )
        print(f"  rasterized {pdf.name} -> {png.name}")


def render_tikz_figure(index: int, figure_body: str) -> Path:
    TIKZ_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"tikz_figure_{index:02d}"
    tex_path = TIKZ_DIR / f"{stem}.tex"
    pdf_path = TIKZ_DIR / f"{stem}.pdf"
    png_path = TIKZ_DIR / f"{stem}.png"

    standalone = rf"""
\documentclass[tikz,border=4pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\begin{{document}}
{figure_body}
\end{{document}}
"""
    tex_path.write_text(standalone, encoding="utf-8")

    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ],
        cwd=TIKZ_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "220",
            "-singlefile",
            str(pdf_path),
            str(TIKZ_DIR / stem),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return png_path


def replace_tikz_figures(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        figure_tex = match.group(0)
        index = repl.counter
        repl.counter += 1
        had_resizebox = "\\resizebox" in figure_tex

        caption_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", figure_tex, flags=re.DOTALL)
        label_match = re.search(r"\\label\{([^}]*)\}", figure_tex)

        content = figure_tex
        content = re.sub(r"^\\begin\{figure\}\[[^\]]*\]\s*", "", content, count=1, flags=re.DOTALL)
        content = re.sub(r"^\\centering\s*", "", content, count=1, flags=re.DOTALL)
        content = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\s*", "", content, count=1, flags=re.DOTALL)
        content = re.sub(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}.*$", "", content, count=1, flags=re.DOTALL)
        content = re.sub(r"\\label\{[^}]*\}", "", content, flags=re.DOTALL)
        content = content.replace("\\end{figure}", "").strip()
        if had_resizebox:
            content = re.sub(r"\}\s*$", "", content, count=1, flags=re.DOTALL)

        png_path = render_tikz_figure(index, content)
        rel_path = png_path.relative_to(PAPER_DIR).as_posix()
        caption = caption_match.group(1).strip() if caption_match else ""
        label = f"\\label{{{label_match.group(1)}}}\n" if label_match else ""

        return (
            "\n\\begin{figure}[H]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.95\\linewidth]{{{rel_path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"{label}"
            "\\end{figure}\n"
        )

    repl.counter = 1
    return re.sub(
        r"\\begin\{figure\}\[[^\]]*\].*?\\end\{figure\}",
        lambda m: repl(m) if "tikzpicture" in m.group(0) else m.group(0),
        body,
        flags=re.DOTALL,
    )


def preprocess_tex() -> str:
    text = SRC_TEX.read_text(encoding="utf-8")

    # Extract MDPI metadata.
    def grab(cmd: str) -> str:
        m = re.search(r"\\" + cmd + r"\{((?:[^{}]|\{[^{}]*\})*)\}", text, flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    title = grab("Title")
    author = grab("Author")
    abstract = grab("abstract")
    keywords = grab("keyword")
    address = grab("address")
    corres = grab("corres")

    # Strip the original MDPI preamble entirely; rebuild as standard article.
    body_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, flags=re.DOTALL)
    if not body_match:
        raise RuntimeError("Could not locate document body")
    body = body_match.group(1)

    # Remove MDPI-only commands inside the body that pandoc cannot parse.
    mdpi_body_cmds = [
        "authorcontributions",
        "funding",
        "acknowledgments",
        "conflictsofinterest",
        "dataavailability",
        "institutionalreview",
        "informedconsent",
        "sampleavailability",
        "supplementary",
        "abbreviations",
        "reftitle",
        "externalbibliography",
    ]
    for cmd in mdpi_body_cmds:
        body = re.sub(
            r"\\" + cmd + r"\{((?:[^{}]|\{[^{}]*\})*)\}",
            lambda m, c=cmd: f"\n\n\\section*{{{c.replace('_', ' ').title()}}}\n{m.group(1)}\n",
            body,
            flags=re.DOTALL,
        )

    # Replace inline tikz figures with rendered PNGs so docx keeps them.
    body = replace_tikz_figures(body)

    # Replace .pdf includegraphics with .png from figures_png.
    body = re.sub(
        r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\.pdf\}",
        lambda m: f"\\includegraphics{m.group(1) or ''}{{figures_png/{Path(m.group(2)).name}.png}}",
        body,
    )

    # Drop \resizebox wrappers that pandoc mishandles.
    body = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?", "", body)

    preamble = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{cleveref}
\graphicspath{{figures_png/}{figures/}}
"""

    title_block = f"""
\\title{{{title}}}
\\author{{{author}\\\\
\\small {address}\\\\
\\small {corres}}}
\\date{{}}
"""

    front = f"""
\\maketitle
\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\noindent\\textbf{{Keywords:}} {keywords}

"""

    new_tex = preamble + title_block + "\n\\begin{document}\n" + front + body + "\n\\end{document}\n"
    TMP_TEX.write_text(new_tex, encoding="utf-8")
    return str(TMP_TEX)


def build_docx() -> None:
    print("Rasterizing PDF figures...")
    rasterize_pdfs()
    print("Preprocessing LaTeX source...")
    tex_path = preprocess_tex()
    print("Running pandoc -> docx...")
    pypandoc.convert_file(
        tex_path,
        "docx",
        outputfile=str(OUT_DOCX),
        extra_args=[
            f"--resource-path={PAPER_DIR}:{PAPER_DIR / 'figures_png'}:{PAPER_DIR / 'figures'}",
            "--standalone",
        ],
    )
    print(f"Wrote {OUT_DOCX} ({OUT_DOCX.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build_docx()
