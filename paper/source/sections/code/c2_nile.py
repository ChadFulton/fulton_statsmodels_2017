from common import *

# Local level model
nile = sm.datasets.nile.load_pandas().data['volume']
nile.index = pd.date_range('1871', '1970', freq='AS')


def output():
    fig, ax = plt.subplots(figsize=(13, 3), dpi=300)
    ax.plot(nile.index, nile, label='Annual flow volume')
    ax.legend()
    ax.yaxis.grid()
    fig.savefig(os.path.join(PNG_PATH, 'fig_2-nile.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_2-nile.pdf'), dpi=300)

if __name__ == '__main__':
    output()