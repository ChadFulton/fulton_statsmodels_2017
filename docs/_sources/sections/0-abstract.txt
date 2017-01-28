.. Abstract
.. Citation: :cite:``
.. Footnote: [#]_ 


.. .. _`abstract`:

Abstract
--------

This paper describes an object oriented approach to the estimation of time
series models using state space methods and presents an implementation in the
Python programming language. This approach at once allows for fast computation,
a variety of out-of-the-box features, and easy extensibility. We show how to
construct a custom state space model, retrieve filtered and smoothed estimates
of the unobserved state, and perform parameter estimation using classical and
Bayesian methods. The mapping from theory to implementation is presented
explicitly and is illustrated at each step by the development of three example
models: an ARMA(1,1) model, the local level model, and a simple real
business cycle macroeconomic model. Finally, four fully implemented time
series models are presented: SARIMAX, VARMAX, unobserved components, and
dynamic factor models. These models can immediately be applied by users. [#]_

.. [#] I thank Josef Perktold for many helpful discussions. Financial support
       from the Google Summer of Code program and the University of Oregon
       Kleinsorge Fellowship, Department of Economics, is gratefully
       acknowledged.