"""contains utility methods"""

import numpy as np
import pandas as pd


def get_cosmic_data(alpha, met_read):
    mass_stars = np.max(pd.read_hdf('alpha_{}/{}/merger_dat.h5'.format(alpha, met_read), key='mass_stars'))[0]
    BBH = pd.read_hdf('alpha_{}/{}/merger_dat.h5'.format(alpha, met_read), key='mergers')
    return BBH, mass_stars

