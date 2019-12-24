# Overview

---

The `py_grama` package is an implementation of a *grammar of model analysis*
(*grama*)---a language for describing and analyzing models.

## What Models?

Statisticians often use "model" to refer to random variable models. Scientists
and engineers often use "model" to refer to simplified physics resulting in
function models. In *grama* we refer to a collection of random variables and
functions *together* as a model.

## Why *grama*?

Considering both the functional mapping between variables and the uncertainties
in those variables is of critical importance to a full understanding of a given
problem. Given the "split" perspective between statisticians and engineers,
unifying the perspectives is a conceptual challenge.

While much effort in the [uncertainty
quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) (UQ)
community has been made on merging the two perspectives on the *algorithmic*
side, relatively little work has been done to merge the two perspectives
*conceptually*. The aforementioned understanding of "models"---functions plus
random variables---is a step towards conceptually unifying these two
perspectives.

## Why `py_grama`?

Furthermore, virtually *no* work has been done to make UQ techniques easily
learnable and accessible. The `py_grama` package is heavily inspired by the
[Tidyverse](https://www.tidyverse.org/), partly in terms of functional
programming patterns, but primarily in terms of its *user-first perspective*.
`py_grama` is designed to help users learn and use UQ tools to analyze models.

## Why quantify uncertainty?

Uncertainty quantification is a relatively new scientific discipline, so the
motivation for doing UQ may not be immediately obvious. The following example notebooks demonstrate UQ in a number of settings:

- [structural safety: cable design example](https://github.com/zdelrosario/py_grama/blob/master/examples/tension/tension.ipynb)---failing to account for uncertainty can lead to unsafe structures. UQ enables safer design.

## What does it look like?

For a quick demonstration of `py_grama`, see the following demo notebooks:

- The [model building demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/builder_demo.ipynb) shows how to build a *grama* model in a scientifically-reproducible way.
- The [model analysis demo](https://github.com/zdelrosario/py_grama/blob/master/examples/demo/analysis_demo.ipynb) shows how *grama* can be used to analyze an existing model, using compact syntax to probe how both functions and randomness affect model outputs.
