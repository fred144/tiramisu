import numpy as np
import pandas as pd


def unpack_pop_ii_data(
    path: str,
    lum_scaling=1e-5,
    lum_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
    table_column_idx=1,
    return_ids=False,
    return_z=False,
):
    r"""
    Depends on the lookup table function.
    given path or link, gives you look up table luminosities and cleans them up
    sample: https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat
    Parameters
    ----------
    path
        path to file
    lum_scaling
        scaling factor for luminosity, see stsci tables
    lum_link
        link to the lookup table, can be file path or url to csv

    Returns
    -------
    star_positions
        (x,y,z) positions of stars
    scaled_stellar_lums
        corresponding stellar luminosities
    masses
        masses in M_sun
    ages

    t_myr
        current time in Myr
    """

    pop_2_data = np.loadtxt(path)
    # birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:, 1] * 1e6  # convert to myr
    ages[ages < 1e6] = 1e6  # set minimum age
    t_myr = pop_2_data[0, 6]  # current simulation time
    z = pop_2_data[1, 6]
    masses = pop_2_data[:, 5]  # msun

    # use look up table; current bottle neck
    stellar_lums = lum_look_up_table(
        stellar_ages=ages, table_link=lum_link, column_idx=table_column_idx, log=True
    )

    scaled_stellar_lums = stellar_lums * lum_scaling
    star_positions = pop_2_data[:, 2:5]  # (x,y,z)

    if return_ids is True:
        if return_z is True:
            return (
                star_positions,
                scaled_stellar_lums,
                masses,
                ages,
                (t_myr, z),
                pop_2_data[:, 0],
            )
        else:
            return (
                star_positions,
                scaled_stellar_lums,
                masses,
                ages,
                t_myr,
                pop_2_data[:, 0],
            )
    else:
        if return_z is True:
            return star_positions, scaled_stellar_lums, masses, ages, (t_myr, z)
        else:
            return star_positions, scaled_stellar_lums, masses, ages, t_myr
