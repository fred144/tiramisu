import numpy as np
import pandas as pd
import os


def lum_look_up_table(
    stellar_ages: float,
    stellar_masses=10,
    table_link: str = os.path.join("..", "starburst", "l1500_inst_e.txt"),
    column_idx: int = 1,
    log=False,
    m_gal=1e6,
):
    """

    given stsci link and ages, returns likely (log) luminosities
    does this via residuals
    Here are some tables.
    https://www.stsci.edu/science/starburst99/docs/table-index.html
    Data File Format:
    Column 1 : Time [yr]
    Column 2 : Solid Line
    Column 3 : Long Dashed Line
    Column 4 : Short Dashed Line

    M = 10^6 M_sun
    Mlow = 1 M_sun

    Solid line:
    alpha = 2.35, Mup = 100 M

    Long-dashed line:
    alpha = 3.30, Mup = 100 M

    Short-dashed line:
    alpha = 2.35, Mup = 30 M


    Parameters
    ----------
    stellar_ages : float
        ages fo the stars in years
    table_link : str
        link, either URL or filepath to the table
    column_idx : int
        column index to use for the tables
    log : TYPE, optional
        return log10 luminosities? The default is False.
    m_gal : TYPE, optional
        mass of the galaxy [Msun] from the starburst model. Default is 10^6 Msun

    Returns
    -------
    luminosities : array
        returns the luminosity of the individual stars, default UV luminosity

    """

    if "www" in table_link:
        df = pd.read_csv(table_link, delim_whitespace=True, header=None)
        data = df.to_numpy().astype(float)
    else:
        data = np.loadtxt(table_link)
    look_up_times = data[:, 0]  # yr

    if log is True:
        look_up_lumi = data[:, column_idx]
    else:
        look_up_lumi = 10 ** data[:, column_idx]

    # vectorized but need big memoery requirement for big array
    # residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis])
    # closest_match_idxs = np.argmin(residuals, axis=1)
    # luminosities = look_up_lumi[closest_match_idxs]

    # loop, helps with memory allocation
    ages_mask = np.ones(stellar_ages.size)
    for i, a in enumerate(stellar_ages):
        closest_age_idx = np.argmin(np.abs(look_up_times - a))
        ages_mask[i] = closest_age_idx
    luminosities = look_up_lumi[np.array(ages_mask, dtype="int")]

    if log is True:
        lum_scaled = luminosities + np.log10(stellar_masses / m_gal)
    else:
        lum_scaled = luminosities * (stellar_masses / m_gal)

    return lum_scaled


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
