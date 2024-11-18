import sys

sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np
from tools.cosmo import t_myr_from_z, z_from_t_myr
from matplotlib import cm
import matplotlib
from scipy import interpolate
from tools import plotstyle
import os
from tools.fscanner import filter_snapshots
import h5py as h5
from matplotlib import colors
import matplotlib as mpl
import cmasher as cmr
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def read_outflow_rates(path, start, stop, step=1, boundary="ISM"):
    fpaths, snums = filter_snapshots(
        # "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
        # 306,
        # 466,
        path,
        start,
        stop,
        sampling=step,
        str_snaps=True,
        snapshot_type="pop2_processed",
    )

    mass_outflow = []
    metalmass_outflow = []
    mass_inflow = []
    metalmass_inflow = []
    times = []
    
    if boundary == "ISM":
        for i, file in enumerate(fpaths):
            f = h5.File(file, "r")
            mass_outflow.append(f["Winds/MassOutFlowRate"][()])
            metalmass_outflow.append(f["Winds/MetalMassOutFlowRate"][()] * 3.81)
            mass_inflow.append(f["Winds/MassInFlowRate"][()])
            metalmass_inflow.append(f["Winds/MetalMassInFlowRate"][()] * 3.81)
            times.append(f["Header/time"][()])
    elif boundary == "CGM":
        for i, file in enumerate(fpaths):
            f = h5.File(file, "r")
            # print(file)
            mass_outflow.append(f["HaloWinds/MassOutFlowRate"][()])
            metalmass_outflow.append(f["HaloWinds/MetalMassOutFlowRate"][()] * 3.81)
            mass_inflow.append(f["HaloWinds/MassInFlowRate"][()])
            metalmass_inflow.append(f["HaloWinds/MetalMassInFlowRate"][()] * 3.81)
            times.append(f["Header/time"][()])
    else:
        raise ValueError("boundary must be 'ISM' or 'CGM'")

    f.close()
    return (
        np.array(times),
        np.array(mass_outflow),
        np.array(metalmass_outflow),
        np.array(mass_inflow),
        np.array(metalmass_inflow),
    )
