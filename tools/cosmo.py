"""
cosmology calculators
"""
import sys

sys.path.append("../..")
import numpy as np
import os
import scipy.stats as st
import yt
from yt.funcs import mylog
from yt.utilities.cosmology import Cosmology
import warnings


# reduces some of the outputs when reading in yt data
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def code_age_to_myr(
    all_star_ages, hubble_const, unique_age=True, true_age=False
):
    r"""
    Returns an array with unique birth epochs in Myr given
    raw_birth_epochs = ad['star', 'particle_birth_epoch']
    AND
    hubble = ds.hubble_constant
    Youngest is 0 Myr, all others are relative to the youngest.

    Relative ages option is currently yielding inconsistent results
    """
    cgs_yr = 3.1556926e7  # 1yr (in s)
    cgs_pc = 3.08567758e18  # pc (in cm)
    h_0 = hubble_const * 100  # hubble parameter (km/s/Mpc)
    h_0_invsec = (
        h_0 * 1e5 / (1e6 * cgs_pc)
    )  # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]

    if unique_age is True:
        # process to unique birth epochs only as well as sort them
        be_star_processed = np.array(
            sorted(list(set(all_star_ages.to_ndarray())))
        )
        star_age_myr = (
            be_star_processed * h_0inv_yr
        ) / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()
    else:
        all_stars = all_star_ages
        star_age_myr = all_stars * h_0inv_yr / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()
    if true_age is True:
        return star_age_myr  # + 13.787 * 1e3
    else:
        return relative_ages  # t = 0 is the age of


def t_myr_from_z(z):
    """
    The times are in reasonable agreement, within 1 Myr, deviations due
    to the used value of parameters, which change somewhat over cosmic
    time
    """
    co = Cosmology(
        hubble_constant=0.70,
        omega_matter=0.270000010728836,
        omega_radiation=0.0,
        omega_lambda=0.730000019073486,
    )
    t_myr = np.array(co.t_from_z(z).in_units("Myr"))

    return t_myr


def z_from_t_myr(t_myr):
    """
    The times are in reasonable agreement, within 1 Myr, deviations due
    to the used value of parameters, which change somewhat over cosmic
    time
    """
    co = Cosmology(
        hubble_constant=0.70,
        omega_matter=0.270000010728836,
        omega_radiation=0.0,
        omega_lambda=0.730000019073486,
    )
    z = np.array(co.z_from_t(t_myr * 1e6 * 3.154 * 1e7))

    return z
