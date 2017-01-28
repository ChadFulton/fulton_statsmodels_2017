from common import *
from c2_inf import inf
from c4_inf import ARMA11

from scipy.stats import multivariate_normal, invgamma

def draw_posterior_phi(model, states, sigma2):
    Z = states[0:1, 1:]
    X = states[0:1, :-1]

    tmp = np.linalg.inv(sigma2 * np.eye(1) + np.dot(X, X.T))
    post_mean = np.dot(tmp, np.dot(X, Z.T))
    post_var = tmp * sigma2

    return multivariate_normal(post_mean, post_var).rvs()

def draw_posterior_sigma2(model, states, phi):
    resid = states[0, 1:] - phi * states[0, :-1]
    post_shape = 3 + model.nobs
    post_scale = 3 + np.sum(resid**2)

    return invgamma(post_shape, scale=post_scale).rvs()

np.random.seed(SEED)

from scipy.stats import norm, uniform
from statsmodels.tsa.statespace.tools import is_invertible

# Create the model for likelihood evaluation and the simulation smoother
model = ARMA11(inf)
sim_smoother = model.simulation_smoother()

# Create the random walk and comparison random variables
rw_proposal = norm(scale=0.3)

# Create storage arrays for the traces
n_iterations = 10000
trace = np.zeros((n_iterations + 1, 3))
trace_accepts = np.zeros(n_iterations)
trace[0] = [0, 0, 1.]  # Initial values

# Iterations
for s in range(1, n_iterations + 1):
    # 1. Gibbs step: draw the states using the simulation smoother
    model.update(trace[s-1], transformed=True)
    sim_smoother.simulate()
    states = sim_smoother.simulated_state[:, :-1]

    # 2. Gibbs step: draw the autoregressive parameters, and apply
    # rejection sampling to ensure an invertible lag polynomial
    phi = draw_posterior_phi(model, states, trace[s-1, 2])
    while not is_invertible([1, -phi]):
        phi = draw_posterior_phi(model, states, trace[s-1, 2])
    trace[s, 0] = phi

    # 3. Gibbs step: draw the variance parameter
    sigma2 = draw_posterior_sigma2(model, states, phi)
    trace[s, 2] = sigma2

    # 4. Metropolis-step for the moving-average parameter
    theta = trace[s-1, 1]
    proposal = theta + rw_proposal.rvs()
    if proposal > -1 and proposal < 1:
        acceptance_probability = np.exp(
            model.loglike([phi, proposal, sigma2]) -
            model.loglike([phi, theta, sigma2]))

        if acceptance_probability > uniform.rvs():
            theta = proposal
            trace_accepts[s-1] = 1
    trace[s, 1] = theta


def output():
    from scipy.stats import gaussian_kde

    burn = 1000
    thin = 10

    final_trace = trace[burn:][::thin]

    fig, axes = plt.subplots(2, 2, figsize=(13, 5), dpi=300)

    phi_kde = gaussian_kde(final_trace[:, 0])
    theta_kde = gaussian_kde(final_trace[:, 1])
    sigma2_kde = gaussian_kde(final_trace[:, 2])

    axes[0, 0].hist(final_trace[:, 0], bins=20, normed=True, alpha=1)
    X = np.linspace(0.75, 1.0, 5000)
    line, = axes[0, 0].plot(X, phi_kde(X))
    ylim = axes[0, 0].get_ylim()
    vline = axes[0, 0].vlines(final_trace[:, 0].mean(), ylim[0], ylim[1],
                              linewidth=2)
    axes[0, 0].set(title=r'$\phi$')

    axes[0, 1].hist(final_trace[:, 1], bins=20, normed=True, alpha=1)
    X = np.linspace(-0.9, 0.0, 5000)
    axes[0, 1].plot(X, theta_kde(X))
    ylim = axes[0, 1].get_ylim()
    vline = axes[0, 1].vlines(final_trace[:, 1].mean(), ylim[0], ylim[1],
                              linewidth=2)
    axes[0, 1].set(title=r'$\theta$')

    axes[1, 0].hist(final_trace[:, 2], bins=20, normed=True, alpha=1)
    X = np.linspace(4, 8.5, 5000)
    axes[1, 0].plot(X, sigma2_kde(X))
    ylim = axes[1, 0].get_ylim()
    vline = axes[1, 0].vlines(final_trace[:, 2].mean(), ylim[0], ylim[1],
                              linewidth=2)
    axes[1, 0].set(title=r'$\sigma^2$')

    p1 = plt.Rectangle((0, 0), 1, 1, alpha=0.7)
    axes[0, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"],
                      loc='upper left')

    axes[1, 1].plot(final_trace[:, 0], label=r'$\phi$')
    axes[1, 1].plot(final_trace[:, 1], label=r'$\theta$')
    axes[1, 1].plot(final_trace[:, 2], label=r'$\sigma^2$')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].set(title=r'Trace plots')

    fig.savefig(os.path.join(PNG_PATH, 'fig_5-gibbs-posteriors.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-gibbs-posteriors.pdf'), dpi=300)

if __name__ == '__main__':
    output()
