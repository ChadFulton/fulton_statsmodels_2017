from common import *

# ARMA(1, 1) model
from pandas_datareader.data import DataReader
cpi = DataReader('CPIAUCNS', 'fred', start='1971-01', end='2016-12')
cpi.index = pd.DatetimeIndex(cpi.index, freq='MS')
inf = np.log(cpi).resample('QS').mean().diff()[1:] * 400


def output():
    fig, ax = plt.subplots(figsize=(13, 3), dpi=300)
    ax.plot(inf.index, inf, label=r'$\Delta \log CPI$')
    ax.legend(loc='lower left')
    ax.yaxis.grid()
    fig.savefig(os.path.join(PNG_PATH, 'fig_2-inf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_2-inf.pdf'), dpi=300)

if __name__ == '__main__':
    output()