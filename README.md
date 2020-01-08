# py_grama
[![Documentation Status](https://readthedocs.org/projects/py_grama/badge/?version=latest)](https://py_grama.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/zdelrosario/py_grama.png?branch=master)](https://travis-ci.org/zdelrosario/py_grama) [![codecov](https://codecov.io/gh/zdelrosario/py_grama/branch/master/graph/badge.svg)](https://codecov.io/gh/zdelrosario/py_grama)

Implementation of a *grammar of model analysis* (*grama*). See the [documentation](https://py-grama.readthedocs.io/en/latest/) for more info.

**Note**: This is *pre-release software*, contents subject to change!

# Installation
Clone this repo, change directories and run the following to install dependencies. (Note: I recommend [Anaconda](https://www.anaconda.com/distribution/) as a Python distribution; it takes care of most of the dependencies.)

```bash
$ git clone git@github.com:zdelrosario/py_grama.git
$ cd py_grama/
$ pip install -r requirements.txt
$ pip install .
# Check install
# python
> import grama
```

Note that I also use a fork of `dfply` for many of the examples; suggest cloning and adding [dfply](https://github.com/zdelrosario/dfply) as well. (Note that I'm considering making this `dfply` fork a formal dependency.)

# Quick Tour
`py_grama` has tools for both *building* and *analyzing* models. For a quick look at functionality, see the following notebooks:

- [model building demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/builder_demo.ipynb)
- [model analysis demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/analysis_demo.ipynb)

# Tutorials
The [tutorials](https://github.com/zdelrosario/py_grama/tree/master/tutorials) page has educational materials for learning to work with `py_grama`.
