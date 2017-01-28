import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

import matplotlib.pyplot as plt
OUTPUT_PATH = '../output/'
PNG_PATH = '../images/'
PDF_PATH = '../../../build/latex/'
SEED = 17429

if not os.path.exists(PDF_PATH):
    os.makedirs(PDF_PATH)

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
