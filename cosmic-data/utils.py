"""contains utility methods"""

import numpy as np
import pandas as pd


def pessimistic_filter(bpp):
    """

    Parameters
    ----------
    bpp : `pandas.DataFrame`
        contains bpp output of COSMIC/BSE

    Returns
    -------
    bpp_filter : `pandas.DataFrame`
        filtered bpp output which does not contain
        BBH mergers which have progenitors that enter a common
        envelope with the donor on the Hertzsprung Gap
    """
    ce = bpp.loc[bpp.evol_type == 7]

    # allow for both kinds of donors
    ce_donor_1 = ce.loc[ce.RRLO_1 >= 1]
    ce_donor_2 = ce.loc[ce.RRLO_2 >= 1]

    # get bin nums for systems with HG kstar donors
    HG_donor_1_bin_num = ce_donor_1.loc[ce_donor_1.kstar_1.isin([2, 8])].bin_num.unique()
    HG_donor_2_bin_num = ce_donor_2.loc[ce_donor_2.kstar_2.isin([2, 8])].bin_num.unique()

    # filter out those bin_nums
    bpp_filter = bpp.loc[~(bpp.bin_num.isin(HG_donor_1_bin_num))]
    bpp_filter = bpp_filter.loc[~(bpp_filter.bin_num.isin(HG_donor_2_bin_num))]

    return bpp_filter


def get_cosmic_data(alpha, met, met_read):
    if met < 1e-4:
        mass_stars = np.max(pd.read_hdf('alpha_{}/{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_8.5e-05.h5'.format(alpha, met_read), key='mass_stars'))[0]
        bpp = pd.read_hdf('alpha_{}/{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_8.5e-05.h5'.format(alpha, met_read), key='bpp')
    else:
        mass_stars = np.max(pd.read_hdf('alpha_{}/{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_{}.h5'.format(alpha, met_read, met_read), key='mass_stars'))[0]
        bpp = pd.read_hdf('alpha_{}/{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_{}.h5'.format(alpha, met_read, met_read), key='bpp')
    # assume HG donors kill common envelopes
    bpp = pessimistic_filter(bpp)
    BBH_bin_num = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.sep > 0.0)].bin_num
    BBH = bpp.loc[(bpp.bin_num.isin(BBH_bin_num)) & (bpp.evol_type == 3.0) & (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14)]

    return BBH, mass_stars
