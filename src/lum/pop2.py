import numpy as np
import pandas as pd
from tools.cosmo import code_age_to_myr


def get_star_ages(ram_ds, ram_ad, logsfc):
    """
    star's ages in [Myr]
    """
    first_form = np.loadtxt(logsfc, usecols=2).max()
    current_hubble = ram_ds.hubble_constant
    current_time = float(ram_ds.current_time.in_units("Myr"))

    birth_start = np.round_(
        float(ram_ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
    )
    converted_unfiltered = code_age_to_myr(
        ram_ad["star", "particle_birth_epoch"],
        current_hubble,
        unique_age=False,
    )
    birthtime = np.round(converted_unfiltered + birth_start, 3)  #!
    current_ages = np.array(np.round(current_time, 3) - np.round(birthtime, 3))
    return current_ages
