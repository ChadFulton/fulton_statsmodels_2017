.. Out-of-the-box models
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _out-of-the-box-models:

Out-of-the-box models
---------------------

This paper has focused on demonstrating the creation of classes to specify
and estimate arbitrary state space models. However, it is worth noting that
classes implementing state space models for four of the most popular models in
time series analysis are built in. These classes have been created
exactly as described above (e.g. they are all subclasses of
``sm.tsa.statespace.MLEModel``), and can be used directly or even extended with
their own subclasses. The source code is available, so that they also serve as
advanced examples of what can be accomplished in this framework.

Maximum likelihood estimation is available immediately simply by calling the
``fit`` method. Features include the calculation of reasonable starting values,
the use of appropriate parameter transformations, and enhanced results classes.
Bayesian estimation via posterior simulation can be performed as described
in this paper by taking advantage of the ``loglike`` method and the simulation
smoother. Of course the selection of priors, parameter blocking, etc. must be
manually implemented, as above.

In this section, we briefly describe each time series model and provide
examples.

SARIMAX
^^^^^^^

The seasonal autoregressive integrated moving-average with exogenous regressors
(SARIMAX) model is a generalization of the familiar ARIMA model to allow for
seasonal effects and explanatory variables. It is typically denoted
SARIMAX :math:`(p,d,q)\times(P,D,Q,s)` and can be written as

.. math::

        y_t & = \beta_t x_t + u_t \\
        \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
            \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

where :math:`y_t` is the observed time series and :math:`x_t` are explanatory
regressors. :math:`\phi_p(L), \tilde \phi_P(L^s), \theta_q(L),` and
:math:`\tilde \theta_Q(L^s)` are lag polynomials and :math:`\Delta^d` is the
differencing operator :math:`\Delta`, applied :math:`d` times. This model is
sometimes described as regression with SARIMA errors.

It is straightforward to apply this model to data by creating an instance of
the class ``sm.tsa.SARIMAX``. For example, if we wanted to estimate an
ARMA(1,1) model for US CPI inflation data using this class, the following
code could be used

.. literalinclude:: code/c6_inf.py
   :lines: 4-6

We can also extend this example to take into account annual seasonality.
Below we estimate an SARIMA(1,0,1)x(1,0,1,12) model. This model achieves a
lower value for the Akaike information criterion (AIC), which indicates a
potentially better fit. [#]_

.. literalinclude:: code/c6_inf.py
   :lines: 8-13

.. [#] The Akaike information criterion, as well as several other information
       criteria, is available for all models that extend the
       ``sm.tsa.statespace.MLEModel`` class. See the tables in
       :ref:`appendix-b` for all available attributes and methods.

Unobserved components
^^^^^^^^^^^^^^^^^^^^^

Unobserved components models, also known as structural time series models,
decompose a univariate time series into trend, seasonal, cyclical, and
irregular components. They can be written as:

.. math::

        y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

where :math:`y_t` refers to the observation vector at time :math:`t`,
:math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
seasonal component, :math:`c_t` refers to the cycle, and
:math:`\varepsilon_t` is the irregular. The modeling details of these
components can be found in the package documentation. These models are also
described in depth in Chapter 3 of :cite:`durbin_time_2012`. The class
corresponding to these models is ``sm.tsa.UnobservedComponents``.

As an example, consider extending the model previously applied to the Nile
river data to include a stochastic cycle, as suggested in
:cite:`mendelssohn_stamp_2011`. This is straightforward with the built-in
model; the below example fits the model and plots the unobserved components,
in this case a level and a cycle, in :numref:`figure_6-uc-nile`.

.. literalinclude:: code/c6_nile.py
   :lines: 5-7


.. _figure_6-uc-nile:

.. figure:: images/fig_6-uc-nile.png

   Estimates of the unobserved level and cyclical components of Nile river
   volume.

VAR
^^^

Vector autoregressions are important tools for reduced form time series
analysis of multiple variables. Their form looks similar to an AR(p) model
except that the variables are vectors and the coefficients are matrices.

.. math::

    y_t = \Phi_1 y_{t-1} + \dots + \Phi_p y_{t-p} + \varepsilon_t

These models can be estimated using the ``sm.tsa.VARMAX`` class, which also
allows estimation of vector moving average models and optionally models with
exogenous regressors. [#]_ The following code estimates a vector autoregression
as a state space model (the starting parameters are the OLS estimates) and
generates orthogonalized impulse response functions for shocks to each of the
endogenous variables; these responses are plotted in
:numref:`figure_6-var-irf`. [#]_

.. literalinclude:: code/c6_rbc.py
   :lines: 5-12

.. _figure_6-var-irf:

.. figure:: images/fig_6-var-irf.png

   Impulse response functions derived from a vector autoregression.

.. [#] Estimation of VARMA(p,q) models is practically possible, although it is
       not recommended because no measures are in place to ensure
       identification (for example, the use of Kronecker indices is not yet
       available).

.. [#] Note that the orthogonalization is by Cholesky decomposition, which
       implicitly enforces a a causal ordering to the variables. The order is
       as defined in the provided dataset. Here ``rbc_data`` orders the
       variables as output, labor, consumption.

Dynamic factors
^^^^^^^^^^^^^^^

Dynamic factor models are another set of important reduced form multivariate
models. They can be used to extract a common component from multifarious
data. The general form of the model available here is the so-called static form
of the dynamic factor model and can be written

.. math::

    y_t & = \Lambda f_t + B x_t + u_t \\
    f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + \eta_t \\
    u_t & = C_1 u_{t-1} + \dots + C_1 f_{t-q} + \varepsilon_t

where :math:`y_t` is the endogenous data, :math:`f_t` are the unobserved
factors which follow a vector autoregression, and :math:`x_t` are optional
exogenous regressors. :math:`\eta_t` and :math:`\varepsilon_t` are white noise
error terms, and :math:`u_t` allows the possibility of autoregressive (or
vector autoregressive) errors. In order to identify the factors,
:math:`Var(\eta_t) \equiv I`.

The following code extracts a single factor that
follows an AR(2) process. The error term is not assumed to be autoregressive,
so in this case :math:`u_t = \varepsilon_t`. By default the model assumes the
elements of :math:`\varepsilon_t` are not cross-sectionally correlated (this
assumption can be relaxed if desired). :numref:`figure_6-dfm-irf` plots the
responses of the endogenous variables to an impulse in the unobserved factor.

.. literalinclude:: code/c6_rbc.py
   :lines: 14-20

.. _figure_6-dfm-irf:

.. figure:: images/fig_6-dfm-irf.png

   Impulse response functions derived from a dynamic factor model.

It is often difficult to directly interpret either the filtered estimates of
the unobserved factors or the estimated coefficients of the :math:`\Lambda`
matrix (called the matrix of factor loadings) due to identification issues
related to the factors. For example, notice that
:math:`\Lambda f_t = (-\Lambda) (-f_t)` so that reversing the signs of the
factors and loadings results in an identical model. It is often
informative instead to examine the extent to which each unobserved factor
explains each endogenous variable (see for example
:cite:`jungbacker_likelihood-based_2014`). This can be explored using the
:math:`R^2` value from the regression of each endogenous variable on each
estimated factor and a constant. These values are available in the results
attribute ``coefficients_of_determination``. For the model estimated above, it
is clear that the estimated factor largely tracks output.


.. todo::

    Need to add a note about what was shocked, to each of the models (it's the
    unobserved factor here, and in RBC it's the technology process)