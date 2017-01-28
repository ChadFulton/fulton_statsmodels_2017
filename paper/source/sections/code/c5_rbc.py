from common import *
from c2_rbc import rbc_data, recessions
from cappC_rbc import SimpleRBC

from scipy.stats import truncnorm, norm, invgamma

def draw_posterior_rho(model, states, sigma2, truncate=False):
    Z = states[1:2, 1:]
    X = states[1:2, :-1]

    tmp = 1 / (sigma2 + np.sum(X**2))
    post_mean = tmp * np.squeeze(np.dot(X, Z.T))
    post_var = tmp * sigma2

    if truncate:
        lower = (-1 - post_mean) / post_var**0.5
        upper = (1 - post_mean) / post_var**0.5
        rvs = truncnorm(lower, upper, loc=post_mean, scale=post_var**0.5).rvs()
    else:
        rvs = norm(post_mean, post_var**0.5).rvs()
    return rvs

def draw_posterior_sigma2(model, states, rho):
    resid = states[1, 1:] - rho * states[1, :-1]
    post_shape = 2.00005 + model.nobs
    post_scale = 0.0100005 + np.sum(resid**2)

    return invgamma(post_shape, scale=post_scale).rvs()

np.set_printoptions(suppress=True)
np.random.seed(SEED)

from statsmodels.tsa.statespace.tools import is_invertible
from scipy.stats import multivariate_normal, gamma, invgamma, beta, uniform

# Create the model for likelihood evaluation
calibrated = {
    'disutility_labor': 3.0,
    'depreciation_rate': 0.025,
}
model = SimpleRBC(rbc_data, calibrated=calibrated)
sim_smoother = model.simulation_smoother()

# Specify priors
prior_discount = gamma(6.25, scale=0.04)
prior_cap_share = norm(0.3, scale=0.01)
prior_meas_err = invgamma(2.0025, scale=0.10025)

# Proposals
rw_discount = norm(scale=0.3)
rw_cap_share = norm(scale=0.01)
rw_meas_err = norm(scale=0.003)

# Create storage arrays for the traces
n_iterations = 10000
trace = np.zeros((n_iterations + 1, 7))
trace_accepts = np.zeros((n_iterations, 5))
trace[0] = model.start_params
trace[0, 0] = 100 * ((1 / trace[0, 0]) - 1)

loglike = None

# Iterations
for s in range(1, n_iterations + 1):
    if s % 10000 == 0:
        print s
    # Get the parameters from the trace
    discount_rate = 1 / (1 + (trace[s-1, 0] / 100))
    capital_share = trace[s-1, 1]
    rho = trace[s-1, 2]
    sigma2 = trace[s-1, 3]
    meas_vars = trace[s-1, 4:]**2

    # 1. Gibbs step: draw the states using the simulation smoother
    model.update(np.r_[discount_rate, capital_share, rho, sigma2, meas_vars])
    sim_smoother.simulate()
    states = sim_smoother.simulated_state[:, :-1]

    # 2. Gibbs step: draw the autoregressive parameter, and apply
    # rejection sampling to ensure an invertible lag polynomial
    # In rare cases due to the combinations of other parameters,
    # the mean of the normal posterior will be greater than one
    # and it becomes difficult to draw from a normal distribution
    # even with rejection sampling. In those cases we draw from a
    # truncated normal.
    rho = draw_posterior_rho(model, states, sigma2)
    i = 0
    while rho < -1 or rho > 1:
        if i < 1e2:
            rho = draw_posterior_rho(model, states, sigma2)
        else:
            rho = draw_posterior_rho(model, states, sigma2, truncate=True)
        i += 1
    trace[s, 2] = rho

    # 3. Gibbs step: draw the variance parameter
    sigma2 = draw_posterior_sigma2(model, states, rho)
    trace[s, 3] = sigma2

    # Calculate the loglikelihood
    loglike = model.loglike(np.r_[discount_rate, capital_share, rho, sigma2, meas_vars])

    # 4. Metropolis-step for the discount rate
    discount_param = trace[s-1, 0]
    proposal_param = discount_param + rw_discount.rvs()
    proposal_rate = 1 / (1 + (proposal_param / 100))
    if proposal_rate < 1:
        proposal_loglike = model.loglike(np.r_[proposal_rate, capital_share, rho, sigma2, meas_vars])
        acceptance_probability = np.exp(
            proposal_loglike - loglike +
            prior_discount.logpdf(proposal_param) -
            prior_discount.logpdf(discount_param))

        if acceptance_probability > uniform.rvs():
            discount_param = proposal_param
            discount_rate = proposal_rate
            loglike = proposal_loglike
            trace_accepts[s-1, 0] = 1

    trace[s, 0] = discount_param

    # 5. Metropolis-step for the capital-share
    proposal = capital_share + rw_cap_share.rvs()
    if proposal > 0 and proposal < 1:
        proposal_loglike = model.loglike(np.r_[discount_rate, proposal, rho, sigma2, meas_vars])
        acceptance_probability = np.exp(
            proposal_loglike - loglike +
            prior_cap_share.logpdf(proposal) -
            prior_cap_share.logpdf(capital_share))

        if acceptance_probability > uniform.rvs():
            capital_share = proposal
            trace_accepts[s-1, 1] = 1
            loglike = proposal_loglike
    trace[s, 1] = capital_share

    # 6. Metropolis-step for the measurement errors
    for i in range(3):
        meas_std = meas_vars[i]**0.5
        proposal = meas_std + rw_meas_err.rvs()
        proposal_vars = meas_vars.copy()
        proposal_vars[i] = proposal**2
        if proposal > 0:
            proposal_loglike = model.loglike(np.r_[discount_rate, capital_share, rho, sigma2, proposal_vars])
            acceptance_probability = np.exp(
                proposal_loglike - loglike +
                prior_meas_err.logpdf(proposal) -
                prior_meas_err.logpdf(meas_std))

            if acceptance_probability > uniform.rvs():
                meas_std = proposal
                trace_accepts[s-1, 2+i] = 1
                loglike = proposal_loglike
                meas_vars[i] = proposal_vars[i]
        trace[s, 4+i] = meas_std

from scipy.stats import gaussian_kde

burn = 1000
thin = 10

final_trace = trace.copy()
final_trace = final_trace[burn:][::thin]
final_trace[:, 0] = 1 / (1 + (final_trace[:, 0] / 100))
final_trace[:, 4:] = final_trace[:, 4:]**2

modes = np.zeros(7)
means = np.mean(final_trace, axis=0)
discount_kde = gaussian_kde(final_trace[:, 0])
cap_share_kde = gaussian_kde(final_trace[:, 1])
rho_kde = gaussian_kde(final_trace[:, 2])
sigma2_kde = gaussian_kde(final_trace[:, 3])

# Finish calculating modes
for i in range(3):
    meas_err_kde = gaussian_kde(final_trace[:, 4+i])
    X = np.linspace(np.min(final_trace[:, 4+i]),
                    np.max(final_trace[:, 4+i]), 1000)
    Y = meas_err_kde(X)
    modes[4+i] = X[np.argmax(Y)]

test = pd.DataFrame(final_trace)

print(pd.DataFrame(
    np.c_[modes, means, test.quantile(q=0.05), test.quantile(q=0.95)],
    columns=['Mode', 'Mean', '5 percent', '95 percent']
).to_string(float_format=lambda x: '%.3g' % x))

res = model.smooth(np.median(final_trace, axis=0))
print(res.summary())
gibbs_irfs = res.impulse_responses(40, orthogonalized=True)*100


def output():
    # Trace plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), dpi=300)

    axes[0, 0].hist(final_trace[:, 0], bins=20, normed=True, alpha=1)
    X = np.linspace(0.990, 1.0-1e-4, 1000)
    Y = discount_kde(X)
    modes[0] = X[np.argmax(Y)]
    line, = axes[0, 0].plot(X, Y)
    ylim = axes[0, 0].get_ylim()
    vline = axes[0, 0].vlines(means[0], ylim[0], ylim[1], linewidth=2)
    axes[0, 0].set(title=r'Discount rate $\beta$')

    axes[0, 1].hist(final_trace[:, 1], bins=20, normed=True, alpha=1)
    X = np.linspace(0.280, 0.370, 1000)
    Y = cap_share_kde(X)
    modes[1] = X[np.argmax(Y)]
    axes[0, 1].plot(X, Y)
    ylim = axes[0, 1].get_ylim()
    vline = axes[0, 1].vlines(means[1], ylim[0], ylim[1], linewidth=2)
    axes[0, 1].set(title=r'Capital share $\alpha$')

    axes[1, 0].hist(final_trace[:, 2], bins=20, normed=True, alpha=1)
    X = np.linspace(-0.4, 1-1e-4, 1000)
    Y = rho_kde(X)
    modes[2] = X[np.argmax(Y)]
    axes[1, 0].plot(X, Y)
    ylim = axes[1, 0].get_ylim()
    vline = axes[1, 0].vlines(means[2], ylim[0], ylim[1], linewidth=2)
    axes[1, 0].set(title=r'Technology shock persistence $\rho$')

    axes[1, 1].hist(final_trace[:, 3], bins=20, normed=True, alpha=1)
    X = np.linspace(0.6e-4, 1.4e-4, 1000)
    Y = sigma2_kde(X)
    modes[3] = X[np.argmax(Y)]
    axes[1, 1].plot(X, Y)
    ylim = axes[1, 1].get_ylim()
    vline = axes[1, 1].vlines(means[3], ylim[0], ylim[1], linewidth=2)
    axes[1, 1].ticklabel_format(style='sci', scilimits=(-2, 2))
    axes[1, 1].set(title=r'Technology shock variance $\sigma^2$')

    p1 = plt.Rectangle((0, 0), 1, 1, alpha=0.7)
    axes[0, 0].legend([p1, line, vline],
                      ["Histogram", "Gaussian KDE", "Sample mean"],
                      loc='upper left')

    fig.savefig(os.path.join(PNG_PATH, 'fig_5-rbc-posteriors.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-rbc-posteriors.pdf'), dpi=300)

    # Gibbs IRFs
    fig, ax = plt.subplots(figsize=(13, 2), dpi=300)

    irfs = gibbs_irfs
    lines, = ax.plot(irfs['output'], label='')
    ax.plot(irfs['output'], 'o', label='Output', color=lines.get_color(),
            markersize=4, alpha=0.8)
    lines, = ax.plot(irfs['labor'], label='')
    ax.plot(irfs['labor'], '^', label='Labor', color=lines.get_color(),
            markersize=4, alpha=0.8)
    lines, = ax.plot(irfs['consumption'], label='')
    ax.plot(irfs['consumption'], 's', label='Consumption',
            color=lines.get_color(), markersize=4, alpha=0.8)

    ax.hlines(0, 0, irfs.shape[0], alpha=0.9, linestyle=':', linewidth=1)
    ylim = ax.get_ylim()
    ax.vlines(0, min(ylim[0]+1e-6, -0.1), ylim[1]-1e-6, alpha=0.9,
              linestyle=':', linewidth=1)
    [ax.spines[spine].set(linewidth=0) for spine in ['top', 'right']]
    ax.set(xlabel='Quarters after impulse', ylabel='Impulse response (\%)',
           xlim=(-1, len(irfs)))

    ax.legend(fontsize=15, labelspacing=0.3)

    fig.savefig(os.path.join(PNG_PATH, 'fig_5-gibbs-irf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-gibbs-irf.pdf'), dpi=300)

    # Smoothed states
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(13, 3), dpi=300)

    alpha = 0.1
    q = norm.ppf(1 - alpha / 2)

    capital = res.smoother_results.smoothed_state[0, :]
    capital_se = res.smoother_results.smoothed_state_cov[0, 0, :]**0.5
    capital_lower = capital - capital_se * q
    capital_upper = capital + capital_se * q

    shock = res.smoother_results.smoothed_state[1, :]
    shock_se = res.smoother_results.smoothed_state_cov[1, 1, :]**0.5
    shock_lower = shock - shock_se * q
    shock_upper = shock + shock_se * q

    line_capital, = ax.plot(rbc_data.index, capital, label='Capital')
    ax.fill_between(rbc_data.index, capital_lower, capital_upper, alpha=0.25,
                    color=line_capital.get_color())

    line_shock, = ax.plot(rbc_data.index, shock, label='Technology process')
    ax.fill_between(rbc_data.index, shock_lower, shock_upper, alpha=0.25,
                    color=line_shock.get_color())

    ax.hlines(0, rbc_data.index[0], rbc_data.index[-1], 'k')
    ax.yaxis.grid()

    ylim = ax.get_ylim()
    ax.fill_between(recessions.index, ylim[0]+1e-5, ylim[1]-1e-5, recessions,
                    facecolor='k', alpha=0.1)

    p1 = plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3)
    ax.legend([line_capital, line_shock, p1],
              ["Capital", "Technology process", "NBER recession indicator"],
              loc='lower left')
    fig.savefig(os.path.join(PNG_PATH, 'fig_5-gibbs-states.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_5-gibbs-states.pdf'), dpi=300)

    # Additional trace plots, unused
    # fig, axes = plt.subplots(7, 2, figsize=(13, 16), dpi=300)
    # cumaccepts = (np.cumsum(trace_accepts, axis=0).T /
    #               np.arange(len(trace_accepts)))

    # axes[0, 0].plot(final_trace[:, 2], label=r'$\rho$')
    # axes[1, 0].plot(final_trace[:, 3], label=r'$\sigma^2$')
    # axes[2, 0].plot(final_trace[:, 0], label=r'$\beta$')
    # axes[3, 0].plot(final_trace[:, 1], label=r'$\alpha$')
    # axes[4, 0].plot(final_trace[:, 4], label=r'$\sigma_y^2$')
    # axes[5, 0].plot(final_trace[:, 5], label=r'$\sigma_n^2$')
    # axes[6, 0].plot(final_trace[:, 6], label=r'$\sigma_c^2$')

    # fig.delaxes(axes[0, 1])
    # fig.delaxes(axes[1, 1])
    # axes[2, 1].plot(cumaccepts[0])
    # axes[3, 1].plot(cumaccepts[1])
    # axes[4, 1].plot(cumaccepts[2])
    # axes[5, 1].plot(cumaccepts[3])
    # axes[6, 1].plot(cumaccepts[4]);
    # [axes[i,1].set_ylim(0, 1) for i in range(2, 7)]

if __name__ == '__main__':
    output()
