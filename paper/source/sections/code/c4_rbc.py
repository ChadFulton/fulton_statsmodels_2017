from common import *
from c2_rbc import rbc_data, recessions
from cappC_rbc import SimpleRBC

# Calibrate everything except measurement variances
calibrated = {
    'discount_rate': 0.95,
    'disutility_labor': 3.0,
    'capital_share': 0.36,
    'depreciation_rate': 0.025,
    'technology_shock_persistence': 0.85,
    'technology_shock_var': 0.04**2
}
calibrated_mod = SimpleRBC(rbc_data, calibrated=calibrated)
calibrated_res = calibrated_mod.fit()

calibrated_irfs = calibrated_res.impulse_responses(40, orthogonalized=True) * 100

# Now, estimate the discount rate and the shock parameters
partially_calibrated = {
    'discount_rate': 0.95,
    'disutility_labor': 3.0,
    'capital_share': 0.36,
    'depreciation_rate': 0.025,
}
mod = SimpleRBC(rbc_data, calibrated=partially_calibrated)
res = mod.fit(maxiter=1000)
res = mod.fit(res.params, method='nm', maxiter=1000, disp=False)
print(res.summary())

estimated_irfs = res.impulse_responses(40, orthogonalized=True) * 100


def output():
    # Fully calibrated
    fig, ax = plt.subplots(figsize=(13, 2), dpi=300)

    irfs = calibrated_irfs
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
    ax.vlines(0, ylim[0]+1e-6, ylim[1]-1e-6, alpha=0.9, linestyle=':',
              linewidth=1)
    [ax.spines[spine].set(linewidth=0) for spine in ['top', 'right']]
    ax.set(xlabel='Quarters after impulse', ylabel='Impulse response (\%)',
           xlim=(-1, len(irfs)))

    ax.legend(fontsize=15, labelspacing=0.3)

    fig.savefig(os.path.join(PNG_PATH, 'fig_4-calibrated-irf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_4-calibrated-irf.pdf'), dpi=300)

    # Partially calibrated
    fig, ax = plt.subplots(figsize=(13, 2), dpi=300)

    irfs = estimated_irfs
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

    fig.savefig(os.path.join(PNG_PATH, 'fig_4-estimated-irf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_4-estimated-irf.pdf'), dpi=300)

    # States
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
    fig.savefig(os.path.join(PNG_PATH, 'fig_4-estimated-states.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_4-estimated-states.pdf'), dpi=300)

if __name__ == '__main__':
    output()
