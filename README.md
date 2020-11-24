# py_grama
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02462/status.svg)](https://doi.org/10.21105/joss.02462) [![PyPI version](https://badge.fury.io/py/py-grama.svg)](https://badge.fury.io/py/py-grama) [![Documentation Status](https://readthedocs.org/projects/py_grama/badge/?version=latest)](https://py_grama.readthedocs.io/en/latest/?badge=latest) ![Python package test](https://github.com/zdelrosario/py_grama/workflows/Python%20package%20test/badge.svg) [![codecov](https://codecov.io/gh/zdelrosario/py_grama/branch/master/graph/badge.svg)](https://codecov.io/gh/zdelrosario/py_grama) [![CodeFactor](https://www.codefactor.io/repository/github/zdelrosario/py_grama/badge/master)](https://www.codefactor.io/repository/github/zdelrosario/py_grama/overview/master) 

Implementation of a *grammar of model analysis* (*grama*). See the [documentation](https://py-grama.readthedocs.io/en/latest/) for more info.

# Overview

Grama is a *grammar of model analysis*---a Python package that supports building and analyzing models with quantified uncertainties. This language is heavily inspired by the [Tidyverse](https://www.tidyverse.org/). Grama provides convenient syntax for building a model (with functions and distributions), generating data, and visualizing results. The purpose of this language is to support scientists and engineers learning to handle uncertainty, and to improve documentation + reproducibility of results.

Uncertainty Quantification (UQ) is the science of analyzing uncertainty in scientific problems and using those results to inform decisions. UQ has important applications to building safety-critical engineering systems, and to making high-consequence choices based on scientific models. However, UQ is generally not taught at the undergraduate level: Many engineers leave their undergraduate training with a purely deterministic view of their discipline, which can lead to probabilistic design errors that [negatively impact safety](https://arc.aiaa.org/doi/abs/10.2514/6.2020-0414). To that end, Grama is designed to facilitate rapid model analysis, communication of results, and the teaching of concepts, all with quantified uncertainties. Intended users of `Grama` are scientists and engineers at the undergraduate level and upward, seeking to analyze computationally-lightweight models.

# Installation
Quick install:

```bash
$ pip install py-grama
```

For a manual install clone this repo, change directories and run the following to install dependencies. (Note: I recommend [Anaconda](https://www.anaconda.com/distribution/) as a Python distribution; it takes care of most of the dependencies.)

```bash
$ git clone git@github.com:zdelrosario/py_grama.git
$ cd py_grama/
$ pip install -r requirements.txt
$ pip install .
```

Run the following to check your install:

```bash
$ python
> import grama
```

# Quick Tour
`py_grama` has tools for both *building* and *analyzing* models. For a quick look at functionality, see the following notebooks:

- [video demo](https://youtu.be/jhyB-jQ7EC8)
- [model building demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/builder_demo.ipynb)
- [model analysis demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/analysis_demo.ipynb)

# Tutorials
The [tutorials](https://github.com/zdelrosario/py_grama/tree/master/tutorials) page has educational materials for learning to work with `py_grama`.

# Support and Contributing
If you are seeking support or want to contribute, please see [Contributing](https://github.com/zdelrosario/py_grama/blob/master/contributing.md).
