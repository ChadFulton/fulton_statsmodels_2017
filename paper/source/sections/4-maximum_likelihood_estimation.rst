.. Introduction
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _maximum-likelihood-estimation:

Maximum Likelihood Estimation
-----------------------------

Classical estimation of parameters in state space models is possible because
the likelihood is a byproduct of the filtering recursions. Given a set of
initial parameters, numerical maximization techniques, often quasi-Newton
methods, can be applied to find the set of parameters that maximize (locally)
the likelihood function, :math:`\mathcal{L}(Y_n \mid \psi)`. In this section we
describe how to apply maximum likelihood estimation (MLE) to state space models
in Python. First we show how to apply a minimization algorithm in SciPy to
maximize the likelihood, using the ``loglike`` method. Second, we show how the underlying Statsmodels functionality inherited by our subclasses can be used to
greatly streamline estimation.

In particular, models extending from the ``sm.tsa.statespace.MLEModel``
("``MLEModel``") class can painlessly perform maximum likelihood estimation via
a ``fit`` method. In addition, summary tables, postestimation results, and
model diagnostics are available. :ref:`appendix-b` describes all of the
methods and attributes that are available to subclasses of ``MLEModel``
and to results objects.

Direct approach
^^^^^^^^^^^^^^^

Numerical optimziation routines in Python are available through the Python
package SciPy (:cite:`jones_scipy:_2001`). Generically, these are in the
form of minimizers that accept a function and a set of starting parameters and
return the set of parameters that (locally) minimize the function. There are a
number of available algorithms, including the popular BFGS
(Broyden–Fletcher–Goldfarb–Shannon) method. As is usual when minimization
routines are available, in order to maximize the (log) likelihood, we minimize
its negative.

The code below demonstrates how to apply maximum likelihood estimation to the
``LocalLevel`` class defined in the previous section for the Nile dataset. In
this case, because we have not bothered to define good starting parameters, we
use the Nelder-Mead algorithm that can be more robust than BFGS although it
may converge more slowly.

.. literalinclude:: code/c4_nile.py
   :lines: 5-17

The maximizing parameters are very close to those reported by
:cite:`durbin_time_2012` and achieve a negligibly higher loglikelihood
(-632.53769 versus -632.53770).

Integration with Statsmodels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While likelihood maximization itself can be performed relatively easily, in
practice there are often many other desired quantities aside from just the
optimal parameters. For example, inference often requires measures of parameter
uncertainty (standard errors and confidence intervals). Another issue that
arises is that it is most convenient to allow the numerical optimizer to choose
parameters across the entire real line. This means that some combinations of
parameters chosen by the optimizer may lead to an invalid model specification.
It is sometimes possible to use an optimization procedure with constraints
or bounds, but it is almost always easier to allow the optimizer to choose in
an unconstrained way and then to transform the parameters to fit the model. The
implementation of parameter transformations will be discussed at greater length
below.

While there is no barrier to users calculating those quantities or implementing
transformations, the calculations are standard and there is no reason for each
user to implement them separately. Again we turn to the principle of separation
of concerns made possible through the object oriented programming approach,
this time by making use of the tools available in Statsmodels. In particular, a
new method, ``fit``, is available to automatically perform maximum
likelihood estimation using the starting parameters defined in the
``start_params`` attribute (see above) and returns a results object.

The following code further refines the local level model by adding a new
attribute ``param_names`` that augments output with descriptive parameter
names. There is also a new line in the ``update`` method that implements
parameter transformations: the ``params`` vector is replaced with the output
from the ``update`` method of the parent class (``MLEModel``). If the
parameters are not already transformed, the parent ``update`` method calls the
appropriate transformation functions and returns the transformed parameters. In
this class we have not yet defined any transformation functions, so the parent
``update`` method will simply return the parameters it was given. Later we will
improve the class to force the variance parameter to be positive.

.. literalinclude:: code/c4_nile.py
   :lines: 19-38

With this new definition, we can instantiate our model and perform maximum
likelihood estimation. As one feature of the integration with Statsmodels, the
result object has a ``summary`` method that prints a table of results:

.. literalinclude:: code/c4_nile.py
   :lines: 40-46

.. literalinclude:: output/c4_nile.txt
   :lines: 1-

A second feature is the availability of model diagnostics. Test statistics for
tests of the standardized residuals for normality, heteroskedasticity, and
serial correlation are reported at the bottom of the summary output. Diagnostic
plots can also be produced using the ``plot_diagnostics`` method, illustrated
in :numref:`figure_4-diag-nile`. [#]_ Notice that Statsmodels is aware of the
date index of the Nile dataset and uses that information in the summary table
and diagnostic plots.

.. _figure_4-diag-nile:

.. figure:: images/fig_4-diag-nile.png

   Diagnostic plots for standardised residuals after maximum likelihood
   estimation on Nile data.

A third feature is the availability of forecasting (through the
``get_forecasts`` method) and impulse response functions (through the
``impulse_responses`` method). Due to the nature of the local level model these
are uninteresting here, but will be exhibited in the ARMA(1,1) and real
business cycle examples below.

.. [#] See sections 2.12 and 7.5 of :cite:`durbin_time_2012` for a description
       of the standardized residuals and the definitions of the provided
       diagnostic tests.

Parameter transformations
"""""""""""""""""""""""""

As mentioned above, parameter transformations are an important component of
maximum likelihood estimation in a wide variety of cases. For example, in the
local level model above the two estimated parameters are variances, which
cannot theoretically be negative. Although the optimizer avoided the
problematic regions in the above example, that will not always be the case. As
another example, ARMA models are typically assumed to be stationary. This
requires coefficients that permit inversion of the associated lag polynomial.
Parameter transformations can be used to enforce these and other
restrictions.

For example, if an unconstrained variance parameter is squared the transformed
variance parameter will always be positive. :cite:`monahan_note_1984` and
:cite:`ansley_note_1986` describe transformations sufficient to induce
stationarity in the univariate and multivariate cases, respectively, by taking
advantage of the one-to-one correspondence between lag polynomial coefficients
and partial autocorrelations. [#]_

It is strongly preferred that the transformation function have a well-defined
inverse so that starting parameters can be specified in terms of the model
space and then "untransformed" to appropriate values in the unconstrained
space.

Implementing parameter transformations when using ``MLEModel`` as the base
class is as simple as adding two new methods: ``transform_params`` and
``untransform_params`` (if no parameter transformations as required, these
methods can simply be omitted from the class definition). The following code
redefines the local level model again, this time to include parameter
transformations to ensure positive variance parameters. [#]_

.. literalinclude:: code/c4_nile.py
   :lines: 48-73

All of the code given above then applies equally to this new model, except that
this class is robust to the optimizer selecting negative parameters.

.. [#] The transformations to induce stationarity are made available in this
       package as the functions
       ``sm.tsa.statespace.tools.constrain_stationary_univariate`` and
       ``sm.tsa.statespace.tools.constrain_stationary_multivariate``. Their
       inverses are also available.

.. [#] Note that in Python, the exponentiation operator is ``**``.

Example models
^^^^^^^^^^^^^^

In this section, we extend the code from :ref:`representation-in-python` to
allow for maximum likelihood estimation through Statsmodels integration.

.. todo::

    Put a reference in here to the out-of-the-box models / section.

ARMA(1, 1) model
""""""""""""""""

.. literalinclude:: code/c4_inf.py
   :lines: 4-37

The parameters can now be easily estimated via maximum likelihood using the
``fit`` method. This model also allows us to demonstrate the prediction and
forecasting features provided by the Statsmodels integration. In particular, we
use the ``get_prediction`` method to retrieve a prediction object that gives
in-sample one-step-ahead predictions and out-of-sample forecasts, as well as
confidence intervals. :numref:`figure_4-forecast-inf` shows a graph of the
output.

.. literalinclude:: code/c4_inf.py
   :lines: 39-44

.. _figure_4-forecast-inf:

.. figure:: images/fig_4-forecast-inf.png

   In-sample one-step-ahead predictions and out-of-sample forecasts for
   ARMA(1,1) model on US CPI inflation data.

If only out-of-sample forecasts had been desired, the ``get_forecasts``
method could have been used instead, and if only the forecasted values had
been desired (and not additional results like confidence intervals), the
methods ``predict`` or ``forecast`` could have been used.

Local level model
"""""""""""""""""

See the previous sections for the Python implementation of the local level
model.

Real business cycle model
"""""""""""""""""""""""""

Due to the the complexity of the model, the full code for the model is too
long to display inline, but it is provided in the :ref:`appendix-c`. It
implements the real business cycle model in a class named ``SimpleRBC`` and
allows selecting some of the structural parameters to be estimated while
allowing others to be calibrated (set to specific values).

Often in structural models one of the outcomes of interest is the time paths of
the observed variables following a hypothetical structural shock; these time
paths are called impulse response functions, and they can be generated for any
state space model.

In the first application, we will calibrate all of the structural parameters to
the values suggested in :cite:`ruge-murcia_methods_2007` and simply estimate
the measurement error variances (these do not affect the model dynamics or the
impulse responses). Once the model has been estimated, the
``impulse_responses`` method can be used to generate the time paths.

.. todo::

    Put in exactly where the code is.

.. literalinclude:: code/c4_rbc.py
   :lines: 5-17

The calculated impulse responses are displayed in
:numref:`figure_4-calibrated-irf`. By calibrating fewer parameters we can
expand estimation to include some of the structural parameters. For example,
we may consider also estimating the two parameters describing the technology
shock. Implementing this only requires eliminating the last two elements from
the ``calibrated`` dictionary. The impulse responses corresponding to this
second exercise are displayed in :numref:`figure_4-estimated-irf`. [#]_

.. _figure_4-calibrated-irf:

.. figure:: images/fig_4-calibrated-irf.png

   Impulse response functions corresponding to a fully calibrated RBC model.

.. _figure_4-estimated-irf:

.. figure:: images/fig_4-estimated-irf.png

   Impulse response functions corresponding to a partially estimated RBC model.

Recall that the RBC model has three observables, output, labor, and
consumption, and two unobserved states, capital and the technology process. The
Kalman filter provides optimal estimates of these unobserved series at time
:math:`t` based on on all data up to time :math:`t`, and the state smoother
provides optimal estimates based on the full dataset. These can be retrieved
from the results object. :numref:`figure_4-estimated-states` displays the
smoothed state values and confidence intervals for the partially estimated
case.

.. _figure_4-estimated-states:

.. figure:: images/fig_4-estimated-states.png

   Smoothed estimates of capital and the technology process from the partially
   estimated RBC model.


.. [#] We note again that this example is merely by way of illustration; it
       does not represent best-practices for careful RBC estimation.