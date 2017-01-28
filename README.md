State Space Estimation of Time Series Models in Python: Statsmodels
===================================================================

This repository houses the source and Python scripts producing the paper
"State Space Estimation of Time Series Models in Python: Statsmodels".

Paper PDF
---------

A PDF version of the paper can be found in the repository, and also at: https://github.com/ChadFulton/fulton_statsmodels_2017/raw/master/fulton_statsmodels_2017_v1.pdf

Notebooks
---------

There are three Jupyter notebooks with code showing maximum likelihood and
Bayesian estimation of three example models:

- [ARMA(1, 1) model](http://nbviewer.jupyter.org/github/ChadFulton/fulton_statsmodels_2017/blob/master/notebooks/ARMA%281%2C%201%29%20-%20CPI%20Inflation.ipynb)
- [Local level model](http://nbviewer.jupyter.org/github/ChadFulton/fulton_statsmodels_2017/blob/master/notebooks/Local%20Level%20-%20Nile.ipynb)
- [Simple Real Business Cycle model](http://nbviewer.jupyter.org/github/ChadFulton/fulton_statsmodels_2017/blob/master/notebooks/Simple%20RBC.ipynb)

Build
-----

The paper is written using [Sphinx](http://www.sphinx-doc.org/en/stable/). In
particular, see:

- `paper/source` for the reStructuredText files of text
- `paper/source/sections/code` for all of the code that is referenced in the
  text and that produces the output and figures. To run all code and produce
  all output, run `python run_all.py` in that directory.
- `notebooks` for Jupyter notebooks that flesh out the three examples in the
  paper (ARMA(1, 1), local level, and a simple real business cycle model)

To build the paper, in a terminal from the base directory, you must:

```bash
>>> cd paper/source/sections/code
>>> python run_all.py
>>> cd ../../../
>>> make html
>>> make latex
```
