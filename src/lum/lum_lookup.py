import numpy as np
import pandas as pd


def lum_look_up_table(stellar_ages, table_link, column_idx: int, log=True):
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
    """
    if "www" in table_link:
        df = pd.read_csv(table_link, delim_whitespace=True, header=None)
        data = df.to_numpy().astype(float)
    else:
        data = np.loadtxt(table_link)
    look_up_times = data[:, 0]  # yr

    if log is True:
        look_up_lumi = 10 ** data[:, column_idx]
    else:
        look_up_lumi = data[:, column_idx]
    residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis])

    closest_match_idxs = np.argmin(residuals, axis=1)

    luminosities = look_up_lumi[closest_match_idxs]
    return luminosities
