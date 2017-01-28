from common import *
from c2_rbc import rbc_data, recessions
from cappC_rbc import SimpleRBC

model = sm.tsa.VARMAX(rbc_data, order=(1, 0))
results = model.fit()

# Generate impulse response functions; the `impluse` argument is used to
# specify which shock is pulsed.
output_irfs = results.impulse_responses(15, impulse=0, orthogonalized=True) * 100
labor_irfs = results.impulse_responses(15, impulse=1, orthogonalized=True) * 100
consumption_irfs = results.impulse_responses(15, impulse=2, orthogonalized=True) * 100

model = sm.tsa.DynamicFactor(rbc_data, k_factors=1, factor_order=2)
results = model.fit()
print(results.coefficients_of_determination)  # [ 0.957   0.545   0.603 ]

# Because the estimated factor turned out to be inversely related to the
# three variables, we want to consider the negative of the impulse
dfm_irfs = -results.impulse_responses(15, impulse=0, orthogonalized=True) * 100


def output():
    # VAR plots
    fig, axes = plt.subplots(3, 1, figsize=(13, 4), dpi=300)
    var_irfs = [output_irfs, labor_irfs, consumption_irfs]

    shocks = ['Output', 'Labor', 'Consumption']
    for i in range(3):
        irfs = var_irfs[i]
        ax = axes[i]
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
        ax.vlines(0, min(ylim[0]+1e-6, -0.1), ylim[1], alpha=0.9,
                  linestyle=':', linewidth=1)
        [ax.spines[spine].set(linewidth=0) for spine in ['top', 'right']]
        ax.set(ylabel='Impulse to \n %s' % shocks[i], xlim=(-1, len(irfs)))
        ax.locator_params(axis='y', nbins=5)
        if i < 2:
            ax.xaxis.set_ticklabels([])
        else:
            ax.set(xlabel='Quarters after impulse')
        if i == 0:
            ax.legend(fontsize=15, labelspacing=0.3, loc='upper right')

    fig.savefig(os.path.join(PNG_PATH, 'fig_6-var-irf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_6-var-irf.pdf'), dpi=300)

    # Dynamic factors plots
    fig, ax = plt.subplots(figsize=(13, 2), dpi=300)

    irfs = dfm_irfs
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
    ax.vlines(0, -0.1, ylim[1]-1e-6, alpha=0.9, linestyle=':', linewidth=1)
    [ax.spines[spine].set(linewidth=0) for spine in ['top', 'right']]
    ax.set(xlabel='Quarters after impulse', ylabel='Impulse response (\%)',
           xlim=(-1, len(irfs)))

    ax.legend(fontsize=15, labelspacing=0.3)

    fig.savefig(os.path.join(PNG_PATH, 'fig_6-dfm-irf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_6-dfm-irf.pdf'), dpi=300)

if __name__ == '__main__':
    output()
