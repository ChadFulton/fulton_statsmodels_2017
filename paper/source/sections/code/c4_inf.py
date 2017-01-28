from common import *
from c2_inf import inf

from statsmodels.tsa.statespace.tools import (constrain_stationary_univariate,
                                              unconstrain_stationary_univariate)

class ARMA11(sm.tsa.statespace.MLEModel):
    start_params = [0, 0, 1]
    param_names = ['phi', 'theta', 'sigma2']

    def __init__(self, endog):
        super(ARMA11, self).__init__(
            endog, k_states=2, k_posdef=1, initialization='stationary')

        self['design', 0, 0] = 1.
        self['transition', 1, 0] = 1.
        self['selection', 0, 0] = 1.

    def transform_params(self, params):
        phi = constrain_stationary_univariate(params[0:1])
        theta = constrain_stationary_univariate(params[1:2])
        sigma2 = params[2]**2
        return np.r_[phi, theta, sigma2]

    def untransform_params(self, params):
        phi = unconstrain_stationary_univariate(params[0:1])
        theta = unconstrain_stationary_univariate(params[1:2])
        sigma2 = params[2]**0.5
        return np.r_[phi, theta, sigma2]

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(ARMA11, self).update(params, **kwargs)

        self['design', 0, 1] = params[1]
        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[2]

inf_model = ARMA11(inf)
inf_results = inf_model.fit()

inf_forecast = inf_results.get_prediction(start='2005-01-01', end='2020-01-01')
print(inf_forecast.predicted_mean)  # [2005-01-01   2.439005 ...
print(inf_forecast.conf_int())      # [2005-01-01   -2.573556 7.451566 ...


def output():
    fig, ax = plt.subplots(figsize=(13, 3), dpi=300)

    forecast = inf_forecast.predicted_mean
    ci = inf_forecast.conf_int(alpha=0.5)

    ax.fill_between(forecast.ix['2017-01-02':].index, -3, 7, color='grey',
                    alpha=0.15)
    lines, = ax.plot(forecast.index, forecast)
    ax.fill_between(forecast.index, ci['lower CPIAUCNS'], ci['upper CPIAUCNS'],
                    alpha=0.2)

    p1 = plt.Rectangle((0, 0), 1, 1, fc="white")
    p2 = plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3)
    ax.legend([lines, p1, p2], ["Predicted inflation",
                                "In-sample one-step-ahead predictions",
                                "Out-of-sample forecasts"], loc='upper left')
    ax.yaxis.grid()

    fig.savefig(os.path.join(PNG_PATH, 'fig_4-forecast-inf.png'), dpi=300)
    fig.savefig(os.path.join(PDF_PATH, 'fig_4-forecast-inf.pdf'), dpi=300)

if __name__ == '__main__':
    output()
