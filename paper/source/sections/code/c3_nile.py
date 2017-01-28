from common import *
from c2_nile import nile

# Create a new class with parent sm.tsa.statespace.MLEModel
class LocalLevel(sm.tsa.statespace.MLEModel):

    # Define the initial parameter vector; see update() below for a note
    # on the required order of parameter values in the vector
    start_params = [1.0, 1.0]

    # Recall that the constructor (the __init__ method) is
    # always evaluated at the point of object instantiation
    # Here we require a single instantiation argument, the
    # observed dataset, called `endog` here.
    def __init__(self, endog):
        super(LocalLevel, self).__init__(endog, k_states=1)

        # Specify the fixed elements of the state space matrices
        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        # Initialize as approximate diffuse, and "burn" the first
        # loglikelihood value
        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    # Here we define how to update the state space matrices with the
    # parameters. Note that we must include the **kwargs argument
    def update(self, params, **kwargs):
        # Using the parameters in a specific order in the update method
        # implicitly defines the required order of parameters
        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]

# Instantiate a new object
nile_model_1 = LocalLevel(nile)

# Compute the loglikelihood at values specific to the nile model
print(nile_model_1.loglike([15099.0, 1469.1]))  # -632.537695048

# Try computing the loglikelihood with a different set of values; notice that it is different
print(nile_model_1.loglike([10000.0, 1.0]))  # -687.5456216

# Retrieve filtering output
nile_filtered_1 = nile_model_1.filter([15099.0, 1469.1])
# print the filtered estimate of the unobserved level
print(nile_filtered_1.filtered_state[0])         # [ 1103.34065938  ... 798.37029261 ]
print(nile_filtered_1.filtered_state_cov[0, 0])  # [ 14874.41126432  ... 4032.15794181 ]

# Retrieve smoothing output
nile_smoothed_1 = nile_model_1.smooth([15099.0, 1469.1])
# print the smoothed estimate of the unobserved level
print(nile_smoothed_1.smoothed_state[0])         # [ 1107.20389814 ... 798.37029261 ]
print(nile_smoothed_1.smoothed_state_cov[0, 0])  # [ 4015.96493689  ... 4032.15794181 ]

np.random.seed(SEED)
# Retrieve a simulation smoothing object
nile_simsmoother_1 = nile_model_1.simulation_smoother()

# Perform first set of simulation smoothing recursions
nile_simsmoother_1.simulate()
print(nile_simsmoother_1.simulated_state[0, :-1])  # [ 1000.09720165 ... 882.30604412 ]

# Perform second set of simulation smoothing recursions
nile_simsmoother_1.simulate()
print(nile_simsmoother_1.simulated_state[0, :-1])  # [ 1153.62271051 ... 808.43895425 ]

# BEFORE: Perform some simulations with the original parameters
nile_simsmoother_1 = nile_model_1.simulation_smoother()
nile_model_1.update([15099.0, 1469.1])
nile_simsmoother_1.simulate()
# ...

# AFTER: Perform some new simulations with new parameters
nile_model_1.update([10000.0, 1.0])
nile_simsmoother_1.simulate()
# ...

nile_model_1.tolerance = 0
nile_model_1.set_filter_method(filter_univariate=True, filter_collapsed=True)


def output():
    # Local level model
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(nile.index, nile, 'k.', label='Observed data')

    line_filtered, = ax.plot(nile.index, nile_filtered_1.filtered_state[0],
                             label=r'Filtered level $(a_{t|t})$')
    se_filtered = nile_filtered_1.filtered_state_cov[0, 0]**0.5
    alpha = 0.50
    q = norm.ppf(1 - alpha / 2.)
    ci_filtered_upper = nile_filtered_1.filtered_state[0] + se_filtered * q
    ci_filtered_lower = nile_filtered_1.filtered_state[0] - se_filtered * q
    ax.fill_between(nile.index, ci_filtered_lower, ci_filtered_upper,
                    alpha=0.1, color=line_filtered.get_color())

    line_smoothed, = ax.plot(nile.index, nile_smoothed_1.smoothed_state[0],
                             label=r'Smoothed level $(\hat \alpha_t)$')
    se_smoothed = nile_smoothed_1.smoothed_state_cov[0, 0]**0.5
    alpha = 0.50
    q = norm.ppf(1 - alpha / 2.)
    ci_smoothed_upper = nile_smoothed_1.smoothed_state[0] + se_smoothed * q
    ci_smoothed_lower = nile_smoothed_1.smoothed_state[0] - se_smoothed * q
    ax.fill_between(nile.index, ci_smoothed_lower, ci_smoothed_upper,
                    alpha=0.1, color=line_smoothed.get_color())

    np.random.seed(SEED)
    nile_simsmoother_1.simulate()
    line, = ax.plot(nile.index, nile_simsmoother_1.simulated_state[0],
                    alpha=0.3, label=r'Level simulations $(\tilde \alpha_t)$')
    for i in range(1, 10):
        nile_simsmoother_1.simulate()
        ax.plot(nile.index, nile_simsmoother_1.simulated_state[0],
                color=line.get_color(), alpha=0.3)

    ax.legend(loc='lower left', fontsize=15, labelspacing=0.3)
    ax.yaxis.grid()
    fig.savefig(os.path.join(PNG_PATH, 'fig_3-model-nile.png'))
    fig.savefig(os.path.join(PDF_PATH, 'fig_3-model-nile.pdf'))

    # Local level model
    fig, ax = plt.subplots(figsize=(13, 3))

    nile_model_1.update([15099.0, 1469.1])

    np.random.seed(SEED)
    nile_simsmoother_1.simulate()
    line, = ax.plot(nile.index, nile_simsmoother_1.simulated_state[0], '-',
                    alpha=0.5, label=r'BEFORE simulations')
    for i in range(1, 10):
        nile_simsmoother_1.simulate()
        ax.plot(nile.index, nile_simsmoother_1.simulated_state[0], '-',
                color=line.get_color(), alpha=0.5)

    nile_model_1.update([10000.0, 1.0])

    np.random.seed(SEED)
    nile_simsmoother_1.simulate()
    line, = ax.plot(nile.index, nile_simsmoother_1.simulated_state[0], '-',
                    alpha=0.3, label=r'AFTER simulations')
    for i in range(1, 10):
        nile_simsmoother_1.simulate()
        ax.plot(nile.index, nile_simsmoother_1.simulated_state[0], '-',
                color=line.get_color(), alpha=0.4)

    ax.legend(loc='upper right', fontsize=15, labelspacing=0.3)
    ax.yaxis.grid()
    fig.savefig(os.path.join(PNG_PATH, 'fig_3-params-simul-nile.png'))
    fig.savefig(os.path.join(PDF_PATH, 'fig_3-params-simul-nile.pdf'))

if __name__ == '__main__':
    output()
