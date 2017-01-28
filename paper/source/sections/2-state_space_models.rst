.. Introduction
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _state-space-models:

State space models
------------------

The state space representation of a possibly time-varying linear and Gaussian
time series model can be written as

.. math::
   :label: sspace

    y_t &= d_t + Z_t \alpha_t + \varepsilon_t \qquad \qquad & \varepsilon_t \sim N(0, H_t) \\
    \alpha_{t+1} & = c_t + T_t \alpha_t + R_t \eta_t  & \eta_t \sim N(0, Q_t) \\

where :math:`y_t` is observed, so the first equation is called the observation
or measurement equation, and :math:`\alpha_t` is unobserved. The second
equation describes the transition of the unobserved state, and so is called the
transition equation. The dimensions of each of the objects, as well as the name
by which we will refer to them, are given in :numref:`table-sspace`. All
notation in this paper will follow that in :cite:`commandeur_statistical_2011`
and :cite:`durbin_time_2012`.

.. _table-sspace:

.. table:: Elements of state space representation

    +-----------------------+-------------------------------------------+--------------------+
    | Object                | Description                               | Dimensions         |
    +=======================+===========================================+====================+
    | :math:`y_t`           | Observed data                             | :math:`p \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`\alpha_t`      | Unobserved state                          | :math:`m \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`d_t`           | Observation intercept                     | :math:`p \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`Z_t`           | Design matrix                             | :math:`p \times m` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`\varepsilon_t` | Observation disturbance                   | :math:`p \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`H_t`           | Observation disturbance covariance matrix | :math:`p \times p` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`c_t`           | State intercept                           | :math:`m \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`T_t`           | Transition matrix                         | :math:`m \times m` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`R_t`           | Selection matrix                          | :math:`m \times r` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`\eta_t`        | State disturbance                         | :math:`r \times 1` |
    +-----------------------+-------------------------------------------+--------------------+
    | :math:`Q_t`           | State disturbance covariance matrix       | :math:`r \times r` |
    +-----------------------+-------------------------------------------+--------------------+

The model is called time-invariant if only :math:`y_t` and :math:`\alpha_t`
depend on time (so, for example, in a time-invariant model
:math:`Z_t = Z_{t+1} \equiv Z`). In the case of a time-invariant model, we will
drop the time subscripts from all state space representation matrices. Many
important time series models are time-invariant, including ARIMA, VAR,
unobserved components, and dynamic factor models.

Kalman Filter
^^^^^^^^^^^^^

The Kalman filter, as applied to the state space model above, is a recursive
formula running forwards through time (:math:`t = 1, 2, \dots, n`) providing
optimal estimates of the unknown state. [#]_ At time :math:`t`,
the *predicted* quantities are the optimal estimates conditional on
observations up to :math:`t-1`, and the *filtered* quantities are the optimal
estimates conditional on observations up to time :math:`t`. This will be
contrasted below with *smoothed* quantities, which are optimal estimates
conditional on the full sample of observations.

We now define some notation that will be useful below. Define the vector of
all observations up to time :math:`s` as :math:`Y_s = \{ y_1, \dots, y_s \}`.
Then the distribution of the predicted state is
:math:`\alpha_t \mid Y_{t-1} \sim N(a_t, P_t)`, and the distribution of the
filtered state is :math:`\alpha_t \mid Y_{t} \sim N(a_{t|t}, P_{t|t})`.

As shown in, for example, :cite:`durbin_time_2012`, the Kalman filter applied
to the model :eq:`sspace` above yields a recursive formulation. Given
prior estimates :math:`a_t, P_t`, the filter produces optimal filtered and
predicted estimates (:math:`a_{t|t}, P_{t|t}` and :math:`a_{t+1}, P_{t+1}`,
respectively) as follows

.. math::
   :label: kfilter

   v_t & = y_t - Z_t a_t - d_t \qquad \qquad & F_t = Z_t P_t Z_t' + H_t \\
   a_{t|t} & = a_t + P_t Z_t' F_t^{-1} v_t & P_{t|t} = P_t - P_t Z_t' F_t^{-1} Z_t P_t \\
   a_{t+1} & = T_t a_{t|t} + c_t & P_{t+1} = T_t P_{t|t} T_t' + R_t Q_t R_t'

An important byproduct of the Kalman filter iterations is evaluation of the
loglikelihood of the observed data due to the so-called "prediction error
decomposition".

The dimensions of each of the objects, as well as the name
by which we will refer to them, are given in :numref:`table-kfilter`. Also
included in the table is the Kalman gain, which is defined as
:math:`K_t = T_t P_t Z_t' F_t^{-1}`.

.. _table-kfilter:

.. table:: Elements of Kalman filter recursions

    +---------------------+-----------------------------------+--------------------+
    | Object              | Description                       | Dimensions         |
    +=====================+===================================+====================+
    | :math:`a_t`         | Prior state mean                  | :math:`m \times 1` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`P_t`         | Prior state covariance            | :math:`m \times m` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`v_t`         | Forecast error                    | :math:`p \times 1` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`F_t`         | Forecast error covariance matrix  | :math:`p \times p` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`a_{t|t}`     | Filtered state mean               | :math:`m \times 1` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`P_{t|t}`     | Filtered state covariance matrix  | :math:`m \times m` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`a_{t+1}`     | Predicted state mean              | :math:`m \times 1` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`P_{t+1}`     | Predicted state covariance matrix | :math:`m \times m` |
    +---------------------+-----------------------------------+--------------------+
    | :math:`\log L(Y_n)` | Loglikelihood                     | scalar             |
    +---------------------+-----------------------------------+--------------------+
    | :math:`K_t`         | Kalman gain                       | :math:`m \times p` |
    +---------------------+-----------------------------------+--------------------+

.. [#] In this paper, "optimal" can be interpreted in the sense of minimizing
       the mean-squared error of estimation. In chapter 4,
       :cite:`durbin_time_2012` show three other senses in which optimal can be
       defined for this same model.

Initialization
^^^^^^^^^^^^^^

Notice that since the Kalman filter is a recursion, for :math:`t = 2, \dots, n`
the prior state mean and prior state covariance matrix are given as the output
of the previous recursion. For :math:`t = 1`, however, no previous recursion
has been yet applied, and so the mean :math:`a_1` and covariance :math:`P_1` of
the distribution of the initial state :math:`\alpha_1 \sim N(a_1, P_1)` must
be specified. The specification of the distribution of the initial state is
referred to as *initialization*.

There are four methods typically used to initialize the Kalman filter:
(1) if the distribution is known or is otherwise specified, initialize with
the known values; (2) initialize with the unconditional distribution of the
process (this is only applicable to the case of stationary state processes);
\(3) initialize with a diffuse (i.e. infinite variance) distribution;
\(4) initialize with an approximate diffuse distribution, i.e. :math:`a_1 = 0`
and :math:`P_1 = \kappa I` where :math:`\kappa` is some large constant (for
example :math:`\kappa = 10^6`). When the state has multiple elements, a mixture
of these four approaches can be used, as appropriate.

Of course, if options (1) or (2) are available, they are preferred. In the case
that there are non-stationary components with unknown initial distribution,
either (3) or (4) must be employed. While (4) is simple to use,
:cite:`durbin_time_2012` note that "while the device can be useful for
approximate exploratory work, it is not recommended for general use since it
can lead to large rounding errors;" for that reason they recommend using exact
diffuse initialiation. For more about initialization, see
:cite:`koopman_filtering_2003` and chapter 5 of :cite:`durbin_time_2012`.

Note that when diffuse initialization is applied, a number of initial
loglikelihood values are excluded ("burned") when calculating the joint
loglikelihood, as they are considered under the influence of the diffuse
states. In exact diffuse initialization the number of burned periods is
determined in initialization, but in the approximate case it must be specified.
In this case, it is typically set equal to the dimension of the state vector;
it turns out that this often coincides with the value in the exact case.

.. todo::

    Need to mention that it's possible to use MLE to estimate the initial
    conditions.

State and disturbance smoothers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The state and disturbance smoothers, as applied to the state space model above,
are recursive formulas running backwards through time
(:math:`t = n, n-1, \dots, 1`) providing optimal estimates of the unknown
state and disturbance vectors based on the full sample of observations.

As developed in :cite:`koopman_disturbance_1993` and Chapter 4 of
:cite:`durbin_time_2012`, following an application of the Kalman filter
(yielding the predicted and filtered estimates of the state) the smoothing
recursions can be written as (where :math:`L_t = T_t - K_t Z_t`)

.. math::
    :label: smoothers
    
    \hat \alpha_t & = a_t + P_t r_{t-1} \qquad \qquad & V_t = P_t - P_t N_{t-1} P_t \\
    \hat \varepsilon_t & = H_t u_t & Var(\varepsilon_t \mid Y_n) = H_t - H_t (F_t^{-1} + K_t' N_t K_t) H_t \\
    \hat \eta_t & = Q_t R_t' r_t & Var(\eta_t \mid Y_n) = Q_t - Q_t R_t' N_t R_t Q_t \\
    u_t & = F_t^{-1} v_t - K_t' r_t \\
    r_{t-1} & = Z_t' u_t + T_t' r_t & N_{t-1} = Z_t' F_t^{-1} Z_t + L_t' N_t L_t

The dimensions of each of the objects, as well as the name
by which we will refer to them, are given in :numref:`table-smoothers`.

.. _table-smoothers:

.. table:: Elements of state and disturbance smoother recursions

    +-------------------------------------+-----------------------------------------------------+--------------------+
    | Object                              | Description                                         | Dimensions         |
    +=====================================+=====================================================+====================+
    | :math:`\hat \alpha_t`               | Smoothed state mean                                 | :math:`m \times 1` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`V_t`                         | Smoothed state covariance matrix                    | :math:`m \times m` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`\hat \varepsilon_t`          | Smoothed observation disturbance mean               | :math:`p \times 1` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`Var(\varepsilon_t \mid Y_n)` | Smoothed observation disturbance covariance matrix  | :math:`p \times p` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`\hat \eta_t`                 | Smoothed state disturbance mean                     | :math:`m \times 1` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`Var(\eta_t \mid Y_n)`        | Smoothed state disturbance covariance matrix        | :math:`m \times m` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`u_t`                         | Smoothing error                                     | :math:`p \times 1` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`r_{t-1}`                     | Scaled smoothed estimator                           | :math:`m \times 1` |
    +-------------------------------------+-----------------------------------------------------+--------------------+
    | :math:`N_{t-1}`                     | Scaled smoothed estimator covariance matrix         | :math:`m \times m` |
    +-------------------------------------+-----------------------------------------------------+--------------------+

Simulation smoother
^^^^^^^^^^^^^^^^^^^

The simulation smoother, developed in :cite:`durbin_simple_2002` and Chapter 4
of :cite:`durbin_time_2012`, allows drawing samples from the distributions of
the full state and disturbance vectors, conditional on the full sample of
observations. It is an example of a "forwards filtering, backwards sampling"
algorithm because one application of the simulation smoother requires one
application each of the Kalman filter and state / disturbance smoother. An
often-used alternative forwards filtering, backwards sampling algorithm is that
of :cite:`carter_gibbs_1994`.

The output of the simulation smoother is the drawn samples; the dimensions of
each of the objects, as well as the name by which we will refer to them, are
given in :numref:`table-simsmoothers`.

.. _table-simsmoothers:

.. table:: Output of the simulation smoother

    +-------------------------------------+-----------------------------------------+--------------------+
    | Object                              | Description                             | Dimensions         |
    +=====================================+=========================================+====================+
    | :math:`\tilde \alpha_t`             | Simulated state                         | :math:`m \times 1` |
    +-------------------------------------+-----------------------------------------+--------------------+
    | :math:`\tilde \varepsilon_t`        | Simulated observation disturbance       | :math:`p \times 1` |
    +-------------------------------------+-----------------------------------------+--------------------+
    | :math:`\tilde \eta_t`               | Simulated state disturbance             | :math:`m \times 1` |
    +-------------------------------------+-----------------------------------------+--------------------+

Practical considerations
^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of important practical considerations associated with the
implementation of the Kalman filter and smoothers in computer code. Two of the
most important are numerical stability and computational speed; these issues
are briefly described below, but will be revisited when the Python
implementation is discussed.

In the context of the Kalman filter, numerical stability usually refers to the
possibility that the recursive calculations will not maintain the positive
definiteness or symmetry of the various covariance matrices. Numerically stable
routines can be used to mitigate these concerns, for example using linear
solvers rather than matrix inversion. In extreme cases a numerically stable
Kalman filter, the so-called square-root Kalman filter, can be used (see
:cite:`morf_square-root_1975` or chapter 6.3 of :cite:`durbin_time_2012`).

Performance can be an issue because the Kalman filter largely consists
of iterations (loops) and matrix operations, and it is well known that loops
perform poorly in interpreted languages like MATLAB and Python. [#]_
Furthermore, regardless of the high-level programming language used, matrix
operations are usually ultimately performed by the highly optimized BLAS and
LAPACK libraries written in Fortran. For performant code, compiled languages
are preferred, and code should directly call the BLAS and LAPACK libraries
directly when possible, rather than through intermediate functions (for details
on the BLAS and LAPACK libraries, see :cite:`anderson_lapack_1999`).

.. [#] The availability of a just-in-time (JIT) compiler can help with loop
       performance in interpreted languages; one is integrated into MATLAB, and
       the Numba project introduces one into Python.

Additional remarks
^^^^^^^^^^^^^^^^^^

Several additional remarks are merited about the Kalman filter. First, under
certain conditions, for example a time-invariant model, the Kalman filter will
converge, meaning that the predicted state covariance matrix, the forecast
error covariance matrix, and the Kalman gain matrix will all reach steady-state
values after some number of iterations. This can be exploited to improve
performance.

The second remark has to do with missing data. In the case of completely or
partially missing observations, not only can the Kalman filter proceed with
making optimal estimates of the state vector, it can provide optimal
estimates of the missing data.

Third, the state space approach can be used to obtain optimal forecasts and to
explore impulse response functions.

Finally, the state space approach can be used for parameter estimation, either
through classical methods (for example maximum likelihood estimation) or
Bayesian methods (for example posterior simulation via Markov chain Monte
Carlo). This will be described in detail in sections 4 and 5, below.

Example models
^^^^^^^^^^^^^^

As mentioned above, many important time series models can be represented in
state space form. We present three models in detail to use as examples: an
autoregressive moving average (ARMA) model, the local level model, and a simple
real business cycle (RBC) dynamic stochastic general equilibrium (DSGE) model.

In fact, general versions of several time series models have already been
implemented in Statsmodels and are available for use (see
:ref:`out-of-the-box-models` for details).
However, since the goal here is to provide information sufficient for users to
specify and estimate their own custom models, we emphasize the translation of
a model from state space formulation to Python code. Below we present state
space representations mathematically, and in subsequent sections we describe
their representation in Python code.

ARMA(1, 1) model
""""""""""""""""

Autoregressive moving average models are widespread in the time series
literature, so we will assume the reader is familiar with their basic
motivation and theory. Suffice it to say, these models are often successfully
applied to obtain reduced form estimates of the dynamics exhibited by time
series and to produce forecasts. For more details, see any introductory time
series text.

An ARMA(1,1) process (where we suppose :math:`y_t` has already been demeaned)
can be written as

.. math::

    y_t = \phi y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}, \qquad \varepsilon_t \sim N(0, \sigma^2)

It is well known that any autoregressive moving average model can be
represented in state-space form, and furthermore that there are many equivalent
representations. Below we present one possible representation based on
:cite:`hamilton_time_1994`, with the corresponding notation from :eq:`sspace`
given below each matrix.

.. math::
    :label: arma11

    y_t & = \underbrace{\begin{bmatrix} 1 & \theta_1 \end{bmatrix}}_{Z} \underbrace{\begin{bmatrix} \alpha_{1,t} \\ \alpha_{2,t} \end{bmatrix}}_{\alpha_t} \\
    \begin{bmatrix} \alpha_{1,t+1} \\ \alpha_{2,t+1} \end{bmatrix} & = \underbrace{\begin{bmatrix}
        \phi & 0 \\
        1      & 0     \\
    \end{bmatrix}}_{T} \begin{bmatrix} \alpha_{1,t} \\ \alpha_{2,t} \end{bmatrix} +
    \underbrace{\begin{bmatrix} 1 \\ 0 \end{bmatrix}}_{R} \underbrace{\varepsilon_{t+1}}_{\eta_t} \\

One feature of ARMA(p,q) models generally is that if the assumption
of stationarity holds, the Kalman filter can be initialized with the
unconditional distribution of the time series.

As an application of this model, in what follows we will consider applying an
ARMA(1,1) model to inflation (first difference of the logged US consumer price
index). This data can be obtained from the Federal Reserve Economic Database
(FRED) produced by the Federal Reserve Bank of St. Louis. In particular, this
data can be easily obtained using the Python package pandas-datareader. [#]_

.. literalinclude:: code/c2_inf.py
   :lines: 4-7

:numref:`figure_2-inf` shows the resulting time series.

.. _figure_2-inf:

.. figure:: images/fig_2-inf.png

   Time path of US CPI inflation from 1971:Q1 - 2016:Q4.

.. [#] This is for illustration purposes only, since an ARMA(1, 1) model with
       mean zero is not a good model for quarterly CPI inflation.

Local level model
"""""""""""""""""

The local level model generalizes the concept of intercept (i.e. "level") in a
linear regression to be time-varying. Much has been written about this model,
and the second chapter of :cite:`durbin_time_2012` is devoted to it. It can be
written as

.. math::
    :label: llevel

    y_t & = \mu_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma_\varepsilon^2) \\
    \mu_{t+1} & = \mu_t + \eta_t, \qquad \eta_t \sim N(0, \sigma_\eta^2) \\

This is already in state space form, with :math:`Z = T = R = 1`. This model is
not stationary (the unobserved level follows a random walk), and so stationary
initialization of the Kalman filter is impossible. Diffuse initialization,
either approximate or exact, is required.

As an application of this model, in what follows we will consider applying
the local level model to the annual flow volume of the Nile river between 1871
and 1970. This data is freely available from many sources, and is included in
many econometrics analysis packages. Here, we use the data from the
Python package Statsmodels.

.. literalinclude:: code/c2_nile.py
   :lines: 4-5

:numref:`figure_2-nile` shows the resulting time series.

.. _figure_2-nile:

.. figure:: images/fig_2-nile.png

   Annual flow volume of the Nile river 1871 - 1970.

Real business cycle model
"""""""""""""""""""""""""

Linearized models can often be placed into state space form and evaluated
using the Kalman filter. A very simple real business cycle model can be
represented as [#]_

.. math::
    :label: rbc

    \begin{bmatrix} y_t \\ n_t \\ c_t \end{bmatrix} & = \underbrace{\begin{bmatrix}
        \phi_{yk} & \phi_{yz} \\
        \phi_{nk} & \phi_{nz} \\
        \phi_{ck} & \phi_{cz} \\
    \end{bmatrix}}_{Z} \underbrace{\begin{bmatrix} k_t \\ z_t \end{bmatrix}}_{\alpha_t} +
    \underbrace{\begin{bmatrix} \varepsilon_{y,t} \\ \varepsilon_{n,t} \\ \varepsilon_{c,t} \end{bmatrix}}_{\varepsilon_t}, \qquad \varepsilon_t \sim N \left ( \begin{bmatrix} 0 \\ 0 \\ 0\end{bmatrix}, \begin{bmatrix}
        \sigma_{y}^2 & 0 & 0 \\
        0 & \sigma_{n}^2 & 0 \\
        0 & 0 & \sigma_{c}^2 \\
    \end{bmatrix} \right ) \\
    \begin{bmatrix} k_{t+1} \\ z_{t+1} \end{bmatrix} & = \underbrace{\begin{bmatrix}
        T_{kk} & T_{kz} \\
        0      & \rho
    \end{bmatrix}}_{T} \begin{bmatrix} k_t \\ z_t \end{bmatrix} +
    \underbrace{\begin{bmatrix} 0 \\ 1 \end{bmatrix}}_{R}
    \eta_t, \qquad \eta_t \sim N(0, \sigma_z^2)

where :math:`y_t` is output, :math:`n_t` is hours worked, :math:`c_t` is
consumption, :math:`k_t` is capital, and :math:`z_t` is a technology shock
process. In this formulation, output, hours worked, and consumption are
observable whereas the capital stock and technology process are unobserved.
This model can be developed as the linearized output of a fully microfounded
DSGE model, see for example :cite:`ruge-murcia_methods_2007` or
:cite:`dejong_structural_2011`. In the theoretical model, the variables are
assumed to be stationary.

There are six structural parameters of this RBC model:
the discount rate, the marginal disutility of labor, the depreciation rate,
the capital-share of output, the technology shock persistence, and the
technology shock innovation variance. It is important to note that the
reduced form parameters of the state space representation (for example
:math:`\phi_{yk}`) are complicated and non-linear functions of these underlying structural parameters.

The raw observable data can be obtained from FRED, although it must be
transformed to be consistent with the model (for example to induce
stationarity). For an explanation of the datasets used and the transformations,
see either of the two references above.

.. literalinclude:: code/c2_rbc.py
   :lines: 4-27

:numref:`figure_2-rbc` shows the resulting time series.

.. _figure_2-rbc:

.. figure:: images/fig_2-rbc.png

   US output, labor, and consumption time series 1984:Q1 - 2016:Q4.

.. todo::

    Either replace all plots with different styles for different lines or else
    remove the different styles from this plot.

.. todo::

    Do I need to show the underlying
    equilibrium equations? Or something?

.. [#] Note that this simple RBC model is presented for illustration purposes
       and so we aim for brevity and clarity of exposition rather than a
       state-of-the-art description of the economy.

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

In order to accomodate parameter estimation, we need to introduce a couple of
new ideas, since the generic state space model described above considers
matrices with known values. In particular (following the notation in Chapter 7
of :cite:`durbin_time_2012`), suppose that the unknown parameters are
collected into a vector :math:`\psi`. Then each of the state space
representation matrices can be considered as, and written as, a function of the
parameters. For example, to take into account the dependence on the unknown
parameters, we write the design matrix as :math:`Z_t(\psi)`.

The three methods for parameter estimation considered in this paper perform
filtering, smoothing, and / or simulation smoothing iteratively, where each
iteration has a (generally) different set of parameter values. Given this
iterative approach, it is clear that in order to perform parameter estimation
we will need two new elements: first, we must have the mappings from parameter
values to fully specified system matrices; second, the iterations must begin
with some initial parameter values and these must be specified.

The first element has already been introduced in the three examples above,
since the state space matrices were written with known values (such as
:math:`1` and :math:`0`) as well as with unknown parameters (for example
:math:`\phi` in the ARMA(1,1) model). The second element will be described
separately for each of parameter estimation methods, below.
