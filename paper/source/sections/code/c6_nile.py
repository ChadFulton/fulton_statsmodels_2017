from common import *
from c2_nile import nile
from c4_nile import MLELocalLevel

model = sm.tsa.UnobservedComponents(nile, 'llevel', cycle=True, stochastic_cycle=True)
results = model.fit()
fig = results.plot_components(observed=False)


def output():
    fig = results.plot_components(observed=False, figsize=(13, 2.8))
    fig.axes[0].xaxis.set_ticklabels([])

    fig.axes[0].yaxis.grid()
    fig.axes[1].yaxis.grid()

    fig.savefig(os.path.join(PNG_PATH, 'fig_6-uc-nile.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_6-uc-nile.pdf'), dpi=300)

if __name__ == '__main__':
    output()
