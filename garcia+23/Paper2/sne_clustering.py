import sys

sys.path.append("../../")
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from tools.fscanner import filter_snapshots
import glob
from matplotlib import cm
import matplotlib as mpl
from tools.ram_fields import ram_fields
from src.lum.lum_lookup import lum_look_up_table
from src.lum.pop2 import get_star_ages
from yt.fields.api import ValidateParameter
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.patches as patches
from tools.check_path import check_path
import cmasher as cmr
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from matplotlib.ticker import LogFormatter
from tools.cosmo import t_myr_from_z

snaps_cc, snap_strings_cc = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/CC-Fiducial/"),
    400,
    400,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)

snaps_70, snap_strings_70 = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/fs07_refine/"),
    1397,
    1397,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)

snaps_35, snap_strings_35 = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/fs035_ms10/"),
    1261,
    1261,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)


def read_sne(logsfc_path, interuped=False):
    log_sfc = np.loadtxt(logsfc_path)
    if interuped is True:
        log_sfc_1 = np.loadtxt(logsfc_path + "-earlier")
        log_sfc = np.concatenate((log_sfc_1, log_sfc), axis=0)

    redshift = log_sfc[:, 2]

    t_myr = t_myr_from_z(redshift)
    # SNe properties
    # m_ejecta = log_sfc[:, 6]
    # e_thermal_injected = log_sfc[:, 7]
    # ejecta_zsun = log_sfc[:, 8]
    # let's do the accumulation of metals produced
    # mass_in_metals = m_ejecta * ejecta_zsun
    # total_mass_in_metals = np.cumsum(mass_in_metals)
    position = log_sfc[:, 12:16]

    # return t_myr, total_mass_in_metals


# %%
from tools.ram_fields import ram_fields

cell_fields, epf = ram_fields()


fpaths, snums = filter_snapshots(
    os.path.expanduser("~/test_data/CC-Fiducial/"),
    397,
    397,
    sampling=1,
    str_snaps=True,
    snapshot_type="ramses_snapshot",
)

for i, sn in enumerate(fpaths):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snums[i]}.txt"))
    print("# reading in", infofile)
    try:
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()
    except:
        print("having trouble reading snapshot, skipping")
        continue

    x_pos = ad["star", "particle_position_x"].to("pc")
    y_pos = ad["star", "particle_position_y"].to("pc")
    z_pos = ad["star", "particle_position_z"].to("pc")
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    plt_ctr = ds.arr([x_center, y_center, z_center], "pc")

    z = ds.current_redshift
    tmyr = np.round(ds.current_time.in_units("Myr").value, 1)
    path = "~/container_tiramisu/sim_log_files/CC-Fiducial/logSN"
    log_sn = np.loadtxt(os.path.expanduser(path))
    log_sn_1 = np.loadtxt(os.path.expanduser(path + "-earlier"))
    log_sn = np.concatenate((log_sn_1, log_sn), axis=0)
    position = log_sn[:, 13:16]

    sn_tmyr = np.round(t_myr_from_z(log_sn[:, 2]), 1)

    snes_now_mask = sn_tmyr == tmyr
    sne_pos_pc = ds.arr(position[snes_now_mask], "code_length").to("pc") - plt_ctr

    p = yt.ProjectionPlot(
        ds, "z", ("gas", "density"), center=plt_ctr, width=(200, "pc")
    )
    p.annotate_marker(
        [sne_pos_pc.value[:, 0], sne_pos_pc.value[:, 1]],
        coord_system="plot",
    )
    p.annotate_timestamp()
    p.save()
    # fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

    # ax.scatter(position[:, 0][snes_now_mask], position[:, 1][snes_now_mask], s=1)
    # # ax.set(xlim=(-200, 200), ylim=(-200, 200))
    # # ax.set(xlim=(0.49, 0.491), ylim=(0.49, 0.491))
    # plt.show()


# f70_log_sn = np.loadtxt(
#     os.path.expanduser("~/container_tiramisu/sim_log_files/fs07_refine/logSN")
# )
# f35_log_sn = np.loadtxt(
#     os.path.expanduser("~/container_tiramisu/sim_log_files/fs035_ms10/logSN")
# )

# %% check clustering
