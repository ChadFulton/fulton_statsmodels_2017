.. Introduction
.. Citation: :cite:``
.. Footnote: [#]_ 

Introduction
------------

The class of time series models that can be represented in state space form,
allowing parameter estimation and inference, is very broad. Many of the most
widespread reduced form time series models fall into this class, including
autoregressive integrated moving average (ARIMA), vector autoregressions
(VARs), unobserved components (UC), time-varying parameters (TVP),
and dynamic factor (DFM) models. Furthermore, linear (or linearized) structural
models are often amenable to representation in this form, including the
important case of linearized DSGE models. This paper contributes to the
literature on practical results related to the estimation of linear, Gaussian
state space models and the corresponding class of time series models.

The great advantage of representing a time series as a linear, Gaussian state
space model is due to existence of the celebrated Kalman filter
(:cite:`kalman_new_1960`), which at once provides optimal contempraneous
estimates of unobserved state variables and also permits evaluation of the
likelihood of the model. Subsequent developments have produced a range of
smoothers and computational techniques which makes feasible a estimation even
in the case of large datasets and complicated models. Elegant theoretical
results can be developed quite generically and applied to any of the models in
the state space class.

Mirroring this theoretical conservation of effort is the possibility of a
practical conservation: appropriately designed computer programs that perform
estimation and inference can be written generically in terms of the state space
form and then applied to any of models which fall into that class. Not only is
it inefficient for each practitioner to separately implement the same
features, it is unreasonable to expect that everyone devote potentially
large amounts of time to produce high-performance, well-tested computer
programs, particularly when their comparative advantage lies elsewhere. This
paper describes a method for achieving this practical conservation of effort by
making use of so-called object oriented programming, with an accompanying
implementation in the Python programming language. [#]_

Time series analysis by state space methods is present in nearly every
statistical software package, including commercial packages like Stata and
E-views, commercial compuational environments such as MATLAB, and open-source
programming languages including R and gretl. A recent special volume
of the Journal of Statistical Software was devoted to software implementations
of state space models; see :cite:`commandeur_statistical_2011` for the
introductory article and a list of references. This is also not the first
implementation of Kalman filtering and smoothing routines in Python; although
many packages at various stages of development exist, one notable reference is
the PySSM package presented in :cite:`strickland_pyssm:_2014`.

Relative to these libraries, this package has several important features.
First, although several of the libraries mentioned above (including the Python
implementation) use object-oriented techniques in their internal code, this is
the first implementation to emphasize those techniques for users of the
library. As described throughout the paper, this can yield substantial time
saving on the part of users, by providing a unified interface to the state
space model rather than a collection of disparate functions.

Second, it is the first implementation to emphasize interaction with an
existing ecosystem of well-estabilished scientific libraries. Since state space estimation is a component of the larger Statsmodels package
(:cite:`seabold_statsmodels:_2010`), users automatically have available many
other econometric and statistical models and functions (in this way,
Statsmodels is somewhat similar to, for example, Stata). It also has links to
other packages; for example, in section 6 we describe Metropolis-Hastings
posterior simulation using the Python package PyMC.

One practically important manifestation of the tighter integration of
Statsmodels with the Python ecosystem is that this package is easy to install
and does not require the user to compile code themselves (as does for example
PySSM). Furthermore, while PySSM also uses compiled code for the performance
critical filtering and smoothing operations, in this package these routines are
written in a close variant of Python (see below for more details on "Cython").
This means that the underlying code is easier to understand and debug and that
a tighter integration can be achieved between user-code and compiled-code.

Finally, it incorporates recent advances in state space model estimation,
including the collapsed filtering approach of
:cite:`jungbacker_likelihood-based_2014`, and makes available flexible classes
for specifying and estimating four of the most popular time series models:
SARIMAX, unobserved components, VARMAX, and dynamic factor models.

One note is warranted about the Python code presented in this paper. In Python,
most functionality is provided by packages not necessarily loaded by default.
To use these packages in your code, you must first "import" them. In all the
code that follows, we will assume the following imports have already been made

.. literalinclude:: code/common.py
   :lines: 1-3

Any additional imports will be explicitly provided in the example code.
In any code with simulations we assume that the following code has been used to
set the seed for the pseudo-random number generator: ``np.random.seed(17429)``.

The remainder of the paper is as follows. Section 2 gives an overview of the
linear, Gaussian state space model along with the Kalman filter, state smoother,
disturbance smoother, and simulation smoother, and presents several examples
of time series models in state space form. Section 3 describes the
representation in Python of the state space model, and provides sample code for
each of the example models. Sections 4 and 5 describe the estimation of unknown
system parameters by maximum likelihood (MLE) and Markov chain Monte Carlo
(MCMC) methods, respectively, and show the application to the
example models. Up to this point, the paper has been concerned with the
implementation of custom state space models. However Statsmodels also contains
a number of out-of-the-box models and these are described in section 6.
Section 7 concludes. [#]_

.. [#] Among others, the programming environments MATLAB and R also support
       object oriented programming; the implementation described here could
       therefore, in principle, be migrated to those languages.

.. [#] For instructions on the installation of this package, see
       :ref:`appendix-a`. Full documentation for the package is available at
       http://www.statsmodels.org.
