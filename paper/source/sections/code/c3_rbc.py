from common import *

class SimpleRBC(sm.tsa.statespace.MLEModel):

    start_params = [...]

    def __init__(self, endog):
        super(SimpleRBC, self).__init__(
            endog, k_states=2, k_posdef=1, initialization='stationary')

        # Initialize RBC-specific variables, parameters, etc.
        # ...

        # Setup fixed elements of the statespace matrices
        self['selection', 1, 0] = 1

    def solve(self, structural_params):
        # Solve the RBC model
        # ...

    def update(self, params, **kwargs):
        params = super(SimpleRBC, self).update(params, **kwargs)

        # Reconstruct the full parameter vector from the
        # estimated and calibrated parameters
        structural_params = ...
        measurement_variances = ...

        # Solve the model
        design, transition = self.solve(structural_params)

        # Update the statespace representation
        self['design'] = design
        self['obs_cov', 0, 0] = measurement_variances[0]
        self['obs_cov', 1, 1] = measurement_variances[1]
        self['obs_cov', 2, 2] = measurement_variances[2]
        self['transition'] = transition
        self['state_cov', 0, 0] = structural_params[...]
