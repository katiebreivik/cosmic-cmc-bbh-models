"""Contains cosmic star formation histories"""

import numpy as np
from scipy.stats import norm


def mean_metal_z(z):
    """
    Mean (mass-weighted) metallicity as a function of redshift
    From Madau & Fragos 2017, courtesy of Mike Zevin

    Parameters
    ----------
    z : `float or numpy.array`
        redshift

    Returns
    -------
    Z : `float or numpy.array`
        mean metallicity
    """
    Zsun = 0.017

    log_Z_Zsun = 0.153 - 0.074 * z ** 1.34
    Z = 10 ** log_Z_Zsun * Zsun
    return Z


def metal_disp_z(z, Z):
    """
    Gives a weight for each metallicity Z at a redshift of z by assuming
    the metallicities are log-normally distributed about Z, code courtesy
    of Mike Zevin

    Parameters
    ----------
    z : `float`
        redshift for computing metallicity distribution

    Z : `numpy.array`
        array of metallicities

    Returns
    -------
    density : `numpy.array`
        distribution of metallicities at redshift z
    """
    # set free parameters in the model
    Zsun = 0.017  # Solar metallicity
    lowZ = Zsun / 200  # lower bound on metallicity
    highZ = 2 * Zsun  # upper bound on metallicity
    sigmaZ = 0.5  # sigma of the lognormal distribution about the mean metallicity

    log_mean_Z = np.log10(mean_metal_z(z)) - (np.log(10) / 2) * sigmaZ ** 2

    Z_dist = norm(loc=log_mean_Z, scale=sigmaZ)
    Z_dist_above = norm(loc=2 * np.log10(highZ) - log_mean_Z, scale=sigmaZ)
    Z_dist_below = norm(loc=2 * np.log10(lowZ) - log_mean_Z, scale=sigmaZ)

    density = Z_dist.pdf(np.log10(Z)) + Z_dist_above.pdf(np.log10(Z)) + Z_dist_below.pdf(np.log10(Z))

    return density


def get_metallicity_weights(zbins, mets):
    """sets up weights for metallicity distribution in star formation
    history for a given set of redshifts and cosmic data metallicity bins

    Parameters
    ----------
    zbins : `numpy.array`
        array of redshift bins to calculate weights for

    mets : `numpy.array`
        array of metallicities corresponding to metallicity grid
        for cosmic data

    Returns
    -------
    met_weights : `numpy.array`
        array of shape (len(mets), len(zbins)) which provides
        the relative weights for how much star formation occurs at
        each metallicity
    """

    met_weights = np.zeros((len(mets), len(zbins)))
    for met, ii in zip(mets, range(len(mets))):
        for zbin_low, zbin_high, jj in zip(zbins[::-1][1:], zbins[::-1][:-1], range(len(zbins))):
            midz = zbin_low + (zbin_high - zbin_low) / 2
            met_weights[ii, jj] = metal_disp_z(max(zbins) - midz, met)

    # normalize so that at each redshift the metallicity weights sum to one
    for ii in range(len(zbins)):
        if np.sum(met_weights[:, ii]) > 0.0:
            met_weights[:, ii] = met_weights[:, ii] / np.sum(met_weights[:, ii])

    return met_weights


def madau_17(z):
    """
    returns star formation rate as a function of supplied redshift

    Parameters
    ----------
    z : `float or numpy.array`
        redshift

    Returns
    -------
    sfr : `float or numpy.array`
        star formation rate [Msun/yr]
    """

    sfr = 0.01 * (1 + z) ** (2.6) / (1 + ((1 + z) / 3.2) ** 6.2)

    return sfr
