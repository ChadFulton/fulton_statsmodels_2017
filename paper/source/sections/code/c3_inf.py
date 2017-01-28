from common import *
from c2_inf import inf

class ARMA11(sm.tsa.statespace.MLEModel):

    start_params = [0, 0, 1]

    def __init__(self, endog):
        super(ARMA11, self).__init__(
            endog, k_states=2, k_posdef=1, initialization='stationary')

        self['design', 0, 0] = 1.
        self['transition', 1, 0] = 1.
        self['selection', 0, 0] = 1.

    def update(self, params, **kwargs):
        self['design', 0, 1] = params[1]
        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[2]

# Example of instantiating a new object, updating the parameters to the
# starting parameters, and evaluating the loglikelihood
inf_model = ARMA11(inf)
print(inf_model.loglike(inf_model.start_params))  # -2682.72563702
