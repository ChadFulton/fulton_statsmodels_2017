from common import *
from c2_nile import nile
from c4_nile import MLELocalLevel

np.random.seed(SEED)

from scipy.stats import multivariate_normal, invgamma, uniform

# Create the model for likelihood evaluation
model = MLELocalLevel(nile)

# Specify priors
prior_obs = invgamma(3, scale=300)
prior_level = invgamma(3, scale=120)

# Specify the random walk proposal
rw_proposal = multivariate_normal(cov=np.eye(2)*10)

# Create storage arrays for the traces
n_iterations = 10000
trace = np.zeros((n_iterations + 1, 2))
trace_accepts = np.zeros(n_iterations)
trace[0] = [120, 30]  # Initial values

# Iterations
for s in range(1, n_iterations + 1):
    proposed = trace[s-1] + rw_proposal.rvs()

    acceptance_probability = np.exp(
        model.loglike(proposed**2) - model.loglike(trace[s-1]**2) +
        prior_obs.logpdf(proposed[0]) + prior_level.logpdf(proposed[1]) -
        prior_obs.logpdf(trace[s-1, 0]) - prior_level.logpdf(trace[s-1, 1]))

    if acceptance_probability > uniform.rvs():
        trace[s] = proposed
        trace_accepts[s-1] = 1
    else:
        trace[s] = trace[s-1]

np.random.seed(SEED)

import pymc as mc

# Priors as "stochastic" elements
prior_obs = mc.InverseGamma('obs', 3, 300)
prior_level = mc.InverseGamma('level', 3, 120)

# Create the model for likelihood evaluation
model = MLELocalLevel(nile)

# Create the "data" component (stochastic and observed)
@mc.stochastic(dtype=sm.tsa.statespace.MLEModel, observed=True)
def loglikelihood(value=model, obs_std=prior_obs, level_std=prior_level):
    return value.loglike([obs_std**2, level_std**2])

# Create the PyMC model
pymc_model = mc.Model((prior_obs, prior_level, loglikelihood))

# Create a PyMC sample and perform sampling
sampler = mc.MCMC(pymc_model)
sampler.sample(iter=10000, burn=1000, thin=10)


def output():
    from scipy.stats import gaussian_kde

    burn = 1000
    thin = 10

    # Direct plots
    final_trace = trace[1 + burn:][::thin]

    fig, axes = plt.subplots(2, 2, figsize=(13, 5), dpi=300)

    obs_kde = gaussian_kde(final_trace[:, 0]**2)
    level_kde = gaussian_kde(final_trace[:, 1]**2)

    axes[0, 0].hist(final_trace[:, 0]**2, bins=20, normed=True, alpha=1)
    X = np.linspace(10000, 26000, 5000)
    line, = axes[0, 0].plot(X, obs_kde(X))
    ylim = axes[0, 0].get_ylim()
    vline = axes[0, 0].vlines(final_trace[:, 0].mean()**2, ylim[0], ylim[1],
                              linewidth=2)
    axes[0, 0].set(title=r'Observation variance ($\sigma_\varepsilon^2$)')

    axes[1, 0].hist(final_trace[:, 1]**2, bins=20, normed=True, alpha=1)
    X = np.linspace(0, 4000, 5000)
    axes[1, 0].plot(X, level_kde(X))
    ylim = axes[1, 0].get_ylim()
    vline = axes[1, 0].vlines(final_trace[:, 1].mean()**2, ylim[0], ylim[1],
                              linewidth=2)
    axes[1, 0].set(title=r'Level variance ($\sigma_\eta^2$)')

    p1 = plt.Rectangle((0, 0), 1, 1, alpha=0.7)
    axes[0, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"])
    axes[1, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"])

    axes[0, 1].plot(final_trace[:, 0]**2, label=r'$\sigma_\varepsilon^2$')
    axes[0, 1].plot(final_trace[:, 1]**2, label=r'$\sigma_\eta^2$')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set(title=r'Trace plots')

    axes[1, 1].plot(trace_accepts.cumsum() / (np.arange(n_iterations) + 1))
    axes[1, 1].set(ylim=(0, 1), title='Acceptance ratio, cumulative')

    fig.savefig(os.path.join(PNG_PATH, 'fig_5-llevel-posteriors.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-llevel-posteriors.pdf'), dpi=300)

    # PyMC plots
    obs_trace = sampler.trace('obs')
    level_trace = sampler.trace('level')
    final_trace = np.c_[obs_trace.gettrace(), level_trace.gettrace()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 5), dpi=300)

    obs_kde = gaussian_kde(final_trace[:, 0]**2)
    level_kde = gaussian_kde(final_trace[:, 1]**2)

    axes[0, 0].hist(final_trace[:, 0]**2, bins=20, normed=True, alpha=1)
    X = np.linspace(5000, 35000, 5000)
    line, = axes[0, 0].plot(X, obs_kde(X))
    ylim = axes[0, 0].get_ylim()
    vline = axes[0, 0].vlines(final_trace[:, 0].mean()**2, ylim[0], ylim[1],
                              linewidth=2)
    axes[0, 0].set(title=r'Observation variance ($\sigma_\varepsilon^2$)')

    axes[1, 0].hist(final_trace[:, 1]**2, bins=20, normed=True, alpha=1)
    X = np.linspace(0, 10000, 5000)
    axes[1, 0].plot(X, level_kde(X))
    ylim = axes[1, 0].get_ylim()
    vline = axes[1, 0].vlines(final_trace[:, 1].mean()**2, ylim[0], ylim[1],
                              linewidth=2)
    axes[1, 0].set(title=r'Level variance ($\sigma_\eta^2$)')

    p1 = plt.Rectangle((0, 0), 1, 1, alpha=0.7)
    axes[0, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"])
    axes[1, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"])

    axes[0, 1].plot(final_trace[:, 0]**2, label=r'$\sigma_\varepsilon^2$')
    axes[0, 1].plot(final_trace[:, 1]**2, label=r'$\sigma_\eta^2$')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set(title=r'Trace plots')

    fig.delaxes(axes[1, 1])

    fig.savefig(os.path.join(PNG_PATH, 'fig_5-pymc-posteriors.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-pymc-posteriors.pdf'), dpi=300)

if __name__ == '__main__':
    output()
