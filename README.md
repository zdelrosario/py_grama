# py_grama
[![PyPI version](https://badge.fury.io/py/py-grama.svg)](https://badge.fury.io/py/py-grama) [![Documentation Status](https://readthedocs.org/projects/py_grama/badge/?version=latest)](https://py_grama.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/zdelrosario/py_grama.png?branch=master)](https://travis-ci.org/zdelrosario/py_grama) [![codecov](https://codecov.io/gh/zdelrosario/py_grama/branch/master/graph/badge.svg)](https://codecov.io/gh/zdelrosario/py_grama)

Implementation of a *grammar of model analysis* (*grama*). See the [documentation](https://py-grama.readthedocs.io/en/latest/) for more info.

# Overview
Grama is a *grammar of model analysis*---a domain-specific language embedded in Python that supports building and analyzing models with quantified uncertainties. This language is heavily inspired by the [Tidyverse](https://www.tidyverse.org/). Grama provides convenient syntax for building a model (with functions and distributions), generating data, and visualizing results. The purpose of this language is to support scientists and engineers learning to handle uncertainty, and to improve documentation + reproducibility of results.

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
