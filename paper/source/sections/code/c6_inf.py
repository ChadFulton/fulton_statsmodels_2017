from common import *
from c2_inf import inf

model_1 = sm.tsa.SARIMAX(inf, order=(1, 0, 1))
results_1 = model_1.fit()
print(model_1.loglike(results_1.params))  # -432.375194381

model_2 = sm.tsa.SARIMAX(inf, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
results_2 = model_2.fit()

# Compare the two models on the basis of the Akaike information criterion
print(results_1.aic)  # 870.750388763
print(results_2.aic)  # 844.623363003
