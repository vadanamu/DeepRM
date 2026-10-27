"""Sphinx configuration for the DeepRM project."""

# ─── Path so autodoc finds project packages ───────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# ─── Basic project info ───────────────────────────────────────────────────────
project = "DeepRM"
author = "Laboratory of Computational Biology, School of Biological Sciences, Seoul National University"
copyright = "2025-2026, Seoul National University R&DB Foundation and Genome4me Inc."
release = "1.2"

# ─── Extensions ───────────────────────────────────────────────────────────────

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "sphinx_design",
]


html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../images/deeprm.png"

mermaid_version = "10.9.1"

templates_path = ["_templates"]
exclude_patterns = []

# --- Autodoc look & feel ---
autodoc_default_options = {
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # types go to description (more readable)
autodoc_inherit_docstrings = True


# --- Napoleon for NumPy docstrings ---
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_custom_sections = ["notes"]
# Make Napoleon rewrite common short names to canonical targets
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "np.ndarray": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
}

# --- MyST ---
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "substitution"]


# --- Make imports reliable for RTD and local builds ---
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # if using src/ layout
autodoc_mock_imports = ["torch", "pysam", "pod5", "pandas", "sklearn", "scipy", "tqdm", "matplotlib", "flash_attn"]

# -- Intersphinx cross-links (optional) --------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "pandas": ("https://pandas.pydata.org/docs/", {}),
    "torch": ("https://pytorch.org/docs/stable/", {}),  # <-- add this
}


# --- Option names keep their two hyphens ----------------------------------------
# Smart quotes would turn "--method" into an en dash ("–method"). smartquotes_action "qe" keeps quotes and ellipses
# only, on every page; the option lists that sphinx-argparse writes, and the option-group sections around them, are
# also left out of smart quotes entirely.
from docutils import nodes as _nodes
from sphinx.transforms import SphinxTransform as _SphinxTransform

smartquotes_action = "qe"


class _OptionListsKeepHyphens(_SphinxTransform):
    default_priority = 740  # before Sphinx's smart quotes (750)

    def apply(self, **kwargs):
        for option_list in self.document.findall(_nodes.option_list):
            option_list["support_smartquotes"] = False
            if isinstance(option_list.parent, _nodes.section):
                option_list.parent["support_smartquotes"] = False


def setup(app):
    app.add_transform(_OptionListsKeepHyphens)
