# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

project = "heataxis"
author = "Bart R. H. Geurten"
copyright = "2026, Bart R. H. Geurten"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autodoc_mock_imports = ["matplotlib", "scipy"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
