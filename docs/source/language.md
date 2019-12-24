# Language Theory

---

*grama* is a conceptual language, and `py_grama` is a code implementation of
that concept. This page is a description of both the concept and its
implementation.

## Running example

We'll use a running example throughout this page; the built-in `py_grama` Cantilever Beam model.

```python
import grama as gr
from grama.models import make_cantilever_beam

md_beam = make_cantilever_beam()
md_beam.printpretty()
```

```bash
model: Cantilever Beam

  inputs:
    var_det:
      w: [2, 4]
      t: [2, 4]
    var_rand:
      H: (+1) norm, {'loc': 500.0, 'scale': 100.0}
      V: (+1) norm, {'loc': 1000.0, 'scale': 100.0}
      E: (+0) norm, {'loc': 29000000.0, 'scale': 1450000.0}
      Y: (-1) norm, {'loc': 40000.0, 'scale': 2000.0}
  functions:
    cross-sectional area: ['w', 't'] -> ['c_area']
    limit state: stress: ['w', 't', 'H', 'V', 'E', 'Y'] -> ['g_stress']
    limit state: displacement: ['w', 't', 'H', 'V', 'E', 'Y'] -> ['g_disp']
```

## Objects

*grama* considers two categories of objects:

- **data** (`df`): observations on various quantities, implemented by the Python package `Pandas`
- **models** (`md`): a function and complete description of its inputs, implemented by `py_grama`

For readability, we suggest using prefixes `df_` and `md_` when naming DataFrames and models.

Since data is already well-handled by Pandas, `py_grama` focuses on providing tools to handle models. A `py_grama` model has **functions** and **inputs**:  The method `printpretty()` gives a quick summary of the model's inputs and function outputs. Model inputs are organized into:

|            | Deterministic                            | Random     |
| ---------- | ---------------------------------------- | ---------- |
| Variables  | `model.var_det`                          | `model.var_rand` |
| Parameters | `model.density.marginals[i].d_param`     | (Future*)  |

- **Variables** are inputs to the model's functions
  + **Deterministic** variables are chosen by the user; the model above has `w, t`
  + **Random** variables are not controlled; the model above has `H, V, E, Y`
- **Parameters** define random variables
  + **Deterministic** parameters are currently implemented; these are listed under `var_rand` with their associated random variable
  + **Random** parameters* are not yet implemented

The `outputs` section lists the various model outputs. The model above has `c_area, g_stress, g_displ`.

## Verbs

Verbs are used to take action on different *grama* objects. We use verbs to generate data from models, build new models from data, and ultimately make sense of the two.

The following table summarizes the categories of `py_grama` verbs. Verbs take either data (`df`) or a model (`md`), and may return either object type. The prefix of a verb immediately tells one both the input and output types. The short prefix is used to denote the *pipe-enabled version* of a verb.

| Verb Type | Prefix (Short)  | In   | Out   |
| --------- | --------------- | ---- | ----- |
| Evaluate  | `eval_` (`ev_`) | `md` | `df`  |
| Fit       | `fit_`  (`ft_`) | `df` | `md`  |
| Transform | `tran_` (`tf_`) | `df` | `df`  |
| Compose   | `comp_` (`cp_`) | `md` | `md`  |


## Functional Programming (Pipes)

[Functional programming](https://en.wikipedia.org/wiki/Functional_programming)
touches both the practical and conceptual aspects of the language. `py_grama`
provides tools to use functional programming patterns. Short-stem versions of
`py_grama` functions are *pipe-enabled*, meaning they can be used in functional
programming form with the pipe operator `>>`. These pipe-enabled functions are
simply aliases for the base functions, as demonstrated below:

```python
df_base = gr.eval_nominal(md_beam, df_det="nom")
df_functional = md_beam >> gr.ev_nominal(df_det="nom")

df_base.equals(df_functional)
```

```bash
True
```

Functional patterns enable chaining multiple commands, as demonstrated in the Sobol' index code above. In nested form using base functions, this would be:

```python
df_sobol = gr.tran_sobol(gr.eval_hybrid(md_beam, n_samples=1e3, df_det="nom", seed=101))
```

From the code above, it is difficult to see that we first consider `md_beam`, perform a hybrid-point evaluation, then use those data to estimate Sobol' indices. With more chained functions, this only becomes more difficult. One could make the code significantly more readable by introducing intermediate variables:

```python
df_samples = gr.eval_hybrid(md_beam, n_samples=1e3, df_det="nom", seed=101)
df_sobol = gr.tran_sobol(df_samples)
```

Conceptually, using *pipe-enabled* functions allows one to skip assigning intermediate variables, and instead pass results along to the next function. The pipe operator `>>` inserts the results of one function as the first argument of the next function. A pipe-enabled version of the code above would be:

```python
df_sobol = \
    md_beam >> \
    gr.ev_hybrid(n_samples=1e3, df_det="nom", seed=101) >> \
    gr.tf_sobol()
```
