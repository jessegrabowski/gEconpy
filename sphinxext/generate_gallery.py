"""
Sphinx plugin to run generate a gallery for notebooks.

Modified from the pymc project, which modified the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import os
import subprocess

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image
from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

HEAD = """
Example Gallery
===============

.. toctree::
   :hidden:

"""

SECTION_TEMPLATE = """
.. _gallery-{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{doc_name}`
      :img-top: {image}
      :link: {doc_reference}
      :link-type: {link_type}
      :shadow: none
"""

folder_title_map = {
    "introductory": "Introductory",
    "estimation": "Estimation",
    "case_study": "Case Studies",
}


def is_tracked_by_git(filepath):
    """Check if a file is tracked by git."""
    # Git indexes a symlink as one entry and reports nothing for paths that run through it, so resolve first.
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(Path(filepath).resolve())],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        # git not available, assume all files are valid
        return True
    else:
        return result.returncode == 0


def write_placeholder(outfile, title, width=275, height=275):
    """Write a plain thumbnail carrying the notebook name, for notebooks with no image output."""
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1), frameon=False, xticks=[], yticks=[])
    ax.text(0.5, 0.5, title.replace("_", " "), ha="center", va="center", wrap=True, fontsize=14)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrite `infile` with a new file of the given size."""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cy * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes((0, 0, 1, 1), aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator:
    """Tools for generating an example page from a file."""

    def __init__(self, filename, thumbnail_dir):
        self.basename = Path(filename).name
        self.stripped_name = Path(filename).stem
        self.png_path = Path(thumbnail_dir) / f"{self.stripped_name}.png"

        with filename.open(encoding="utf-8") as fid:
            self.json_source = json.load(fid)

    def extract_preview_pic(self):
        """By default, just uses the last image in the notebook."""
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
        return None

    def gen_previews(self):
        if self.png_path.exists():
            logger.info(
                f"Custom thumbnail already exists for {self.basename}, skipping extraction",
                type="thumbnail_extractor",
            )
            return

        preview = self.extract_preview_pic()
        if preview is not None:
            with self.png_path.open("wb") as buff:
                buff.write(preview)
        else:
            logger.info(
                f"Didn't find any pictures in {self.basename}, using a placeholder thumbnail",
                type="thumbnail_extractor",
            )
            write_placeholder(self.png_path, self.stripped_name)
        create_thumbnail(self.png_path)


def main(app):
    logger.info("Starting thumbnail extractor.")

    working_dir = Path.cwd()
    os.chdir(app.builder.srcdir)
    try:
        write_gallery()
    finally:
        os.chdir(working_dir)


def write_gallery():
    """Write examples/gallery.rst and the thumbnails, relative to the Sphinx source directory."""
    toc_entries = []
    sections = []

    for folder, title in folder_title_map.items():
        sections.append(SECTION_TEMPLATE.format(section_title=title, section_id=folder, underlines="-" * len(title)))

        thumbnail_dir = Path("_thumbnails") / folder
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        for nb_path in sorted(Path("examples", folder).glob("*.ipynb")):
            if not is_tracked_by_git(nb_path):
                logger.info(
                    f"Skipping {nb_path.name}, not tracked by git",
                    type="thumbnail_extractor",
                )
                continue

            nbg = NotebookGenerator(filename=nb_path, thumbnail_dir=thumbnail_dir)
            nbg.gen_previews()

            doc_name = f"{folder}/{nbg.stripped_name}"
            toc_entries.append(f"   {doc_name}")
            sections.append(
                ITEM_TEMPLATE.format(
                    doc_name=doc_name,
                    image=f"/{nbg.png_path}",
                    doc_reference=doc_name,
                    link_type="doc",
                )
            )

    with Path("examples", "gallery.rst").open("w", encoding="utf-8") as f:
        f.write(HEAD + "\n".join(toc_entries) + "\n" + "\n".join(sections))


def setup(app):
    app.connect("builder-inited", main)
