.. Appendix A
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _appendix-b:

Appendix B: Inherited attributes and methods
--------------------------------------------

``sm.tsa.statespace.MLEModel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The methods available to all classes inheriting from the base classes
``sm.tsa.statespace.MLEModel`` are listed in
:numref:`table-mlemodel-methods` and the attributes are listed in
:numref:`table-mlemodel-attributes`.

.. _table-mlemodel-methods:

.. table:: Methods available to subclasses of ``sm.tsa.statespace.MLEModel``

    +------------------------------------+------------------------------------------------------------+
    | Method                             | Description                                                |
    +====================================+============================================================+
    | ``filter``                         | Kalman filtering                                           |
    +------------------------------------+------------------------------------------------------------+
    | ``fit``                            | Fits the model by maximum likelihood via Kalman filter.    |
    +------------------------------------+------------------------------------------------------------+
    | ``loglike``                        | Joint loglikelihood evaluation                             |
    +------------------------------------+------------------------------------------------------------+
    | ``loglikeobs``                     | Loglikelihood evaluation                                   |
    +------------------------------------+------------------------------------------------------------+
    | ``set_filter_method``              | Set the filtering method                                   |
    +------------------------------------+------------------------------------------------------------+
    | ``set_inversion_method``           | Set the inversion method                                   |
    +------------------------------------+------------------------------------------------------------+
    | ``set_stability_method``           | Set the numerical stability method                         |
    +------------------------------------+------------------------------------------------------------+
    | ``set_conserve_memory``            | Set the memory conservation method                         |
    +------------------------------------+------------------------------------------------------------+
    | ``set_smoother_output``            | Set the smoother output                                    |
    +------------------------------------+------------------------------------------------------------+
    | ``simulation_smoother``            | Retrieve a simulation smoother for the statespace model.   |
    +------------------------------------+------------------------------------------------------------+
    | ``initialize_known``               | Initialize the Kalman filter with known values             |
    +------------------------------------+------------------------------------------------------------+
    | ``initialize_approximate_diffuse`` | Specify approximate diffuse Kalman filter initialization   |
    +------------------------------------+------------------------------------------------------------+
    | ``initialize_stationary``          | Initialize the statespace model as stationary              |
    +------------------------------------+------------------------------------------------------------+
    | ``simulate``                       | Simulate a new time series following the state space model |
    +------------------------------------+------------------------------------------------------------+
    | ``impulse_responses``              | Impulse response function                                  |
    +------------------------------------+------------------------------------------------------------+

.. _table-mlemodel-attributes:

.. table:: Attributes available to subclasses of ``sm.tsa.statespace.MLEModel``

    +------------------------+---------------------------------------------------------------------------------+
    | Attribute              | Description                                                                     |
    +========================+=================================================================================+
    | ``endog``              | The observed (endogenous) dataset                                               |
    +------------------------+---------------------------------------------------------------------------------+
    | ``exog``               | The dataset of explanatory variables (if applicable)                            |
    +------------------------+---------------------------------------------------------------------------------+
    | ``start_params``       | Parameter vector used to initialize parameter estimation iterations             |
    +------------------------+---------------------------------------------------------------------------------+
    | ``param_names``        | Human-readable names of parameters                                              |
    +------------------------+---------------------------------------------------------------------------------+
    | ``initialization``     | The selected method for Kalman filter initialization                            |
    +------------------------+---------------------------------------------------------------------------------+
    | ``initial_variance``   | The initial variance to use in approximate diffuse initialization               |
    +------------------------+---------------------------------------------------------------------------------+
    | ``loglikelihood_burn`` | The number of observations during which the likelihood is not evaluated         |
    +------------------------+---------------------------------------------------------------------------------+
    | ``tolerance``          | The tolerance at which the Kalman filter determines convergence to steady-state |
    +------------------------+---------------------------------------------------------------------------------+

.. _table-mlemodel-slices:

.. table:: Slice keys available to subclasses of ``sm.tsa.statespace.MLEModel``

    +------------------------+---------------------------------------------------------------------------------+
    | Attribute              | Description                                                                     |
    +========================+=================================================================================+
    | ``'obs_intercept'``    | Observation intercept; :math:`d_t`                                              |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'design'``           | Design matrix; :math:`Z_t`                                                      |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'obs_cov'``          | Observation disturbance covariance matrix; :math:`H_t`                          |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'state_intercept'``  | State intercept; :math:`c_t`                                                    |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'transition'``       | Transition matrix; :math:`T_t`                                                  |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'selection'``        | Selection matrix; :math:`R_t`                                                   |
    +------------------------+---------------------------------------------------------------------------------+
    | ``'state_cov'``        | State disturbance covariance matrix; :math:`Q_t`                                |
    +------------------------+---------------------------------------------------------------------------------+

The ``fit``, ``filter``, and ``smooth`` methods return a
``sm.tsa.statespace.MLEResults`` object; its methods and attributes are given
below.

The ``simulation_smoother`` method returns a ``SimulationSmoothResults``
object; its methods and attributes are also given below.

``sm.tsa.statespace.MLEResults``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The methods available to these results objects are listed in
:numref:`table-mleresults-methods` and the attributes are listed in
:numref:`table-mleresults-attributes`.

.. _table-mleresults-methods:

.. table:: Methods available to results objects from ``fit``, ``filter``, and ``smooth``

    +-----------------------------+------------------------------------------------------------------------------------+
    | Method                      | Description                                                                        |
    +=============================+====================================================================================+
    | ``test_normality``          | Jarque-Bera for normality of standardized residuals.                               |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``test_heteroskedasticity`` | Test for heteroskedasticity (break in the variance) of standardized residuals      |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``test_serial_correlation`` | Ljung-box test for no serial correlation of standardized residuals                 |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``get_prediction``          | In-sample prediction and out-of-sample forecasting; returns all prediction results |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``get_forecast``            | Out-of-sample forecasts; returns all forecasting results                           |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``predict``                 | In-sample prediction and out-of-sample forecasting; only returns predicted values  |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``forecast``                | Out-of-sample forecasts; only returns forecasted values                            |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``simulate``                | Simulate a new time series following the state space model                         |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``impulse_responses``       | Impulse response function                                                          |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``plot_diagnostics``        | Diagnostic plots for standardized residuals of one endogenous variable             |
    +-----------------------------+------------------------------------------------------------------------------------+
    | ``summary``                 | Summarize the results                                                              |
    +-----------------------------+------------------------------------------------------------------------------------+

.. _table-mleresults-attributes:

.. table:: Attributes available to results objects from ``fit``, ``filter``, and ``smooth``

    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | Attribute                                | Description                                                                                 |
    +==========================================+=============================================================================================+
    | ``aic``                                  | Akaike Information Criterion                                                                |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``bic``                                  | Bayes Information Criterion                                                                 |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``bse``                                  | Standard errors of fitted parameters                                                        |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``conf_int``                             | Returns the confidence interval of the fitted parameters                                    |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``cov_params_default``                   | Covariance matrix of fitted parameters                                                      |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``filtered_state``                       | Filtered state mean; :math:`a_{t|t}`                                                        |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``filtered_state_cov``                   | Filtered state covariance matrix; :math:`P_{t|t}`                                           |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``fittedvalues``                         | Fitted values of the model; alias for forecasts.                                            |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``forecasts``                            | Forecasts; :math:`\hat y_t = Z_t a_t`                                                       |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``forecasts_error``                      | Forecast errors; :math:`v_t`                                                                |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``forecasts_error_cov``                  | Forecast error covariance matrix; :math:`F_t`                                               |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``hqic``                                 | Hannan-Quinn Information Criterion                                                          |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``kalman_gain``                          | Kalman gain; :math:`K_t`                                                                    |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``llf_obs``                              | The values of the loglikelihood function at the fitted parameters; :math:`\log L(y_t)`      |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``llf``                                  | The value of the joint loglikelihood function at the fitted parameters; :math:`\log L(Y_n)` |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``loglikelihood_burn``                   | The number of observations during which the likelihood is not evaluated                     |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``nobs``                                 | The number of observations in the dataset                                                   |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``params``                               | The fitted parameters                                                                       |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``predicted_state``                      | Predicted state mean; :math:`a_t`                                                           |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``predicted_state_cov``                  | Predicted state covariance matrix; :math:`P_t`                                              |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``pvalues``                              | The p-values associated with the z-statistics of the coefficients                           |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``resid``                                | Residuals of the model; alias for forecasts_errors                                          |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_measurement_disturbance``     | Smoothed observation disturbance mean; :math:`\hat \varepsilon_t`                           |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_measurement_disturbance_cov`` | Smoothed observation disturbance covariance matrix; :math:`Var(\varepsilon_t \mid Y_n)`     |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_state``                       | Smoothed state mean; :math:`\hat \alpha_t`                                                  |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_state_cov``                   | Smoothed state covariance matrix; :math:`V_t`                                               |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_state_disturbance``           | Smoothed state disturbance mean; :math:`\hat \eta_t`                                        |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``smoothed_state_disturbance_cov``       | Smoothed state disturbance covariance matrix; :math:`Var(\eta_t \mid Y_n)`                  |
    +------------------------------------------+---------------------------------------------------------------------------------------------+
    | ``zvalues``                              | The z-values of the standard errors of fitted parameters                                    |
    +------------------------------------------+---------------------------------------------------------------------------------------------+

``SimulationSmoothResults``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only method of a ``SimulationSmoothResults`` object is given in
:numref:`table-simsmoother-methods`. After this method is called, the
attributes in :numref:`table-simsmoother-attributes` are populated. Each time
the method is called, these attributes change to the newly simulated values.

.. _table-simsmoother-methods:

.. table:: Methods available to results objects from ``simulation_smoother``

    +--------------+------------------------------+
    | Method       | Description                  |
    +==============+==============================+
    | ``simulate`` | Perform simulation smoothing |
    +--------------+------------------------------+

.. _table-simsmoother-attributes:

.. table:: Attributes available to results objects from ``simulation_smoother``

    +---------------------------------------+----------------------------------------------------------------+
    | Attribute                             | Description                                                    |
    +=======================================+================================================================+
    | ``simulated_state``                   | Simulated state vector; :math:`\tilde \alpha_t`                |
    +---------------------------------------+----------------------------------------------------------------+
    | ``simulated_measurement_disturbance`` | Simulated measurment disturbance; :math:`\tilde \varepsilon_t` |
    +---------------------------------------+----------------------------------------------------------------+
    | ``simulated_state_disturbance``       | Simulated state disturbance; :math:`\tilde \eta_t`             |
    +---------------------------------------+----------------------------------------------------------------+
