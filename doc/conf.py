# -*- coding: utf-8 -*-
#
# vispy documentation build configuration file
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
from datetime import date

import sys
import os
import re
import warnings
import vispy

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath('ext'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.6'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.imgmath',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinxcontrib.apidoc',
              'sphinx_gallery.gen_gallery',
              'myst_parser',
              ]

# API docs
apidoc_module_dir = "../vispy"
apidoc_output_dir = "api"
apidoc_excluded_paths = ["../vispy/ext"]
apidoc_separate_modules = True

# Sphinx Gallery

# the following files are ignored from gallery processing
ignore_files = ['plotting/export.py',
                'gloo/geometry_shader.py',
                ]
ignore_pattern_regex = [re.escape(os.sep) + f for f in ignore_files]
ignore_pattern_regex = "|".join(ignore_pattern_regex)

sphinx_gallery_conf = {
    'examples_dirs': ['../examples/gloo', '../examples/scene', '../examples/plotting'],
    'gallery_dirs': ['gallery/gloo', 'gallery/scene', 'gallery/plotting'],
    'filename_pattern': re.escape(os.sep),
    'ignore_pattern': ignore_pattern_regex,
    'only_warn_on_example_error': True,
    'image_scrapers': ('vispy',),
    'reset_modules': tuple(),  # remove default matplotlib/seaborn resetters
    'first_notebook_cell': '%gui qt',  # tell notebooks to use Qt backend
    'within_subsection_order': "FileNameSortKey",
}
# Let vispy.app.application:Application.run know that we are generating gallery images
os.environ["_VISPY_RUNNING_GALLERY_EXAMPLES"] = "1"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The main toctree document.
master_doc = 'index'

# General information about the project.
project = u'VisPy'
copyright = u'2013-{}, VisPy developers'.format(date.today().year)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = vispy.__version__
# The short X.Y version.
version = '.'.join(release.split('.')[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'README.rst']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# Tell the theme where the code lives


def _custom_edit_url(github_user, github_repo, github_version, doc_path, file_name, default_edit_page_url_template):
    """Create custom 'edit' URLs for API modules since they are dynamically generated."""
    if file_name.startswith("api/"):
        # this is a dynamically generated API page, link to actual Python source
        modpath = os.sep.join(os.path.splitext(file_name[4:])[0].split("."))
        if modpath == "modules":
            # main package listing
            modpath = "vispy"
        rel_modpath = os.path.join("..", modpath)
        if os.path.isdir(rel_modpath):
            doc_path = modpath + "/"
            file_name = "__init__.py"
        elif os.path.isfile(rel_modpath + ".py"):
            doc_path = os.path.dirname(modpath)
            file_name = os.path.basename(modpath) + ".py"
        else:
            warnings.warn(f"Not sure how to generate the API URL for: {file_name}")
    return default_edit_page_url_template.format(github_user=github_user,
                                                 github_repo=github_repo,
                                                 github_version=github_version,
                                                 doc_path=doc_path,
                                                 file_name=file_name)


# FIXME: including a function (_custom_edit_url) in html_context causes a warning in the build in CI
# we fail on warnings, so suppress it for now. I think we need to suppress all similar warnings.
# WARNING: cannot cache unpickable configuration value: 'html_context'
# (because it contains a function, class, or module object)
suppress_warnings = ["config.cache"]

html_context = {
    "github_user": "vispy",
    "github_repo": "vispy",
    "github_version": "main",
    "doc_path": "doc",
    "edit_page_url_template": "{{ vispy_custom_edit_url(github_user, github_repo, github_version, doc_path, file_name, default_edit_page_url_template) }}",
    "default_edit_page_url_template": "https://github.com/{github_user}/{github_repo}/edit/{github_version}/{doc_path}{file_name}",
    "vispy_custom_edit_url": _custom_edit_url,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "use_edit_page_button": True,
    "github_url": "https://github.com/vispy/vispy",
    "twitter_url": "https://twitter.com/vispyproject",
    "header_links_before_dropdown": 7,
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'VisPy'

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/vispy-teaser-short.png"


# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {'documentation*': ['localtoc.html', 'searchbox.html'],
#                  'installation*' : ['localtoc.html', 'searchbox.html'],
#                  'modern-gl*'    : ['localtoc.html', 'searchbox.html'],
#                  'api*'          : ['localtoc.html', 'searchbox.html'],
# #                 'resources*'    : ['localtoc.html', 'searchbox.html'],
#                  }

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'vispydoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = []

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = []

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = []

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


# -----------------------------------------------------------------------------
# Numpy extensions
# -----------------------------------------------------------------------------
numpydoc_show_class_members = False


# -----------------------------------------------------------------------------
# intersphinx
# -----------------------------------------------------------------------------
_python_doc_base = "https://docs.python.org/3"
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
}


def setup(app):
    # Add custom CSS
    app.add_css_file('style.css')
