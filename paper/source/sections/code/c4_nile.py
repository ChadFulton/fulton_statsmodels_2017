from common import *
from c2_nile import nile
from c3_nile import LocalLevel

# Load the generic minimization function from scipy
from scipy.optimize import minimize

# Create a new function to return the negative of the loglikelihood
nile_model_2 = LocalLevel(nile)
def neg_loglike(params):
    return -nile_model_2.loglike(params)

# Perform numerical optimization
output = minimize(neg_loglike, nile_model_2.start_params, method='Nelder-Mead')

print(output.x)  # [ 15108.31   1463.55]
print(nile_model_2.loglike(output.x))  # -632.537685587

class FirstMLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(FirstMLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(FirstMLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]

nile_mlemodel_1 = FirstMLELocalLevel(nile)

print(nile_mlemodel_1.loglike([15099.0, 1469.1]))  # -632.537695048

# Again we use Nelder-Mead; now specified as method='nm'
nile_mleresults_1 = nile_mlemodel_1.fit(method='nm', maxiter=1000)
print(nile_mleresults_1.summary())

class MLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(MLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def transform_params(self, params):
        return params**2

    def untransform_params(self, params):
        return params**0.5

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(MLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]


def output():
    with open(os.path.join(OUTPUT_PATH, 'c4_nile.txt'), 'w') as f:
        f.write(str(nile_mleresults_1.summary()))

    fig = nile_mleresults_1.plot_diagnostics(figsize=(13, 5))
    fig.savefig(os.path.join(PNG_PATH, 'fig_4-diag-nile.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_4-diag-nile.pdf'), dpi=300)

if __name__ == '__main__':
    output()
