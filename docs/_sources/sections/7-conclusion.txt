.. Conclusion
.. Citation: :cite:``
.. Footnote: [#]_ 

.. .. _conclusion:

Conclusion
----------

This paper describes the use of the Statsmodels Python library for the
specification and estimation of state space models. It begins by presenting the
notation and equations describing state space models and the filtering,
smoothing, and simulation smoothing operations required for estimation. Next,
it maps these concepts to programming code using the the technique of object
oriented programming and describes a simple method for the specification of
state space models. Brief theoretical introductions to maximum likelihood
estimation and Bayesian posterior simulation are given and mapped to
programming code; the object oriented representation of state space models
makes parameter estimation simple and straightforward.

Three examples, an ARMA(1,1) model, the local level model, and a simple real
business cycle model are developed throughout, first theoretically and then as
models specified in programming code. Classical and Bayesian estimation of the
parameters of each model is performed. Finally, four flexible generic time
series models provided in Statsmodels are described. Using these built-in
classes, two of the example models, the ARMA(1,1) model and the local level
model, are re-estimated and then extended to more complex, better fitting models.
