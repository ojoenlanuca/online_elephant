import numpy as np


def round_to_nearest_fraction_multiple(x, frac):
    return np.round(np.divide(x, frac, dtype=np.float64)) * frac
