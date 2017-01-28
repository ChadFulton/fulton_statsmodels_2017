.. Appendix C
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _appendix-c:

Appendix C: Real business cycle model code
------------------------------------------

This appendix presents Python code implementing the full real business cycle
model, including solution of the linear rational expectations model, as
described in :ref:`representation-in-python`. It also presents code for the
parameter estimation by classical (see :ref:`maximum-likelihood-estimation`)
and Bayesian (see :ref:`posterior_simulation`) methods.

The following code implements the real business cycle model in Python as a
state space model.

.. literalinclude:: code/cappC_rbc.py
   :lines: 3-240

The following code estimates the three measurement variances as well as the
two technology shock parameters via maximum likelihood estimation

.. literalinclude:: code/c4_rbc.py
   :lines: 19-31

Finally, the following code estimates all parameters except the disutility of
labor and the depreciation rate via the Metropolis-within-Gibbs algorithm

.. literalinclude:: code/c5_rbc.py
   :lines: 5-28

.. literalinclude:: code/c5_rbc.py
   :lines: 31-155
