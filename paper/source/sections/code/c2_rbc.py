from common import *

# RBC model
from pandas_datareader.data import DataReader
start = '1984-01'
end = '2016-09'
labor = DataReader('HOANBS', 'fred',start=start, end=end).resample('QS').first()
cons = DataReader('PCECC96', 'fred', start=start, end=end).resample('QS').first()
inv = DataReader('GPDIC1', 'fred', start=start, end=end).resample('QS').first()
pop = DataReader('CNP16OV', 'fred', start=start, end=end)
pop = pop.resample('QS').mean()  # Convert pop from monthly to quarterly observations
recessions = DataReader('USRECQ', 'fred', start=start, end=end)
recessions = recessions.resample('QS').last()['USRECQ'].iloc[1:]

# Get in per-capita terms
N = labor['HOANBS'] * 6e4 / pop['CNP16OV']
C = (cons['PCECC96'] * 1e6 / pop['CNP16OV']) / 4
I = (inv['GPDIC1'] * 1e6 / pop['CNP16OV']) / 4
Y = C + I

# Log, detrend
y = np.log(Y).diff()[1:]
c = np.log(C).diff()[1:]
n = np.log(N).diff()[1:]
i = np.log(I).diff()[1:]
rbc_data = pd.concat((y, n, c), axis=1)
rbc_data.columns = ['output', 'labor', 'consumption']


def output():
    fig, ax = plt.subplots(figsize=(13, 3), dpi=300)

    ax.plot(y.index, y, label=r'Output $(y_t)$')
    ax.plot(n.index, n, label=r'Labor $(n_t)$')
    ax.plot(c.index, c, label=r'Consumption $(c_t)$')

    ax.yaxis.grid()
    ax.legend(loc='lower left', fontsize=15, labelspacing=0.3)
    fig.savefig(os.path.join(PNG_PATH, 'fig_2-rbc.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_2-rbc.pdf'), dpi=300)

if __name__ == '__main__':
    output()