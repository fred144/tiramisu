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

cc_log_sn = np.loadtxt(
    os.path.expanduser("~/container_tiramisu/sim_log_files/CC-Fiducial/logSN")
)
f70_log_sn = np.loadtxt(
    os.path.expanduser("~/container_tiramisu/sim_log_files/fs07_refine/logSN")
)
f35_log_sn = np.loadtxt(
    os.path.expanduser("~/container_tiramisu/sim_log_files/fs035_ms10/logSN")
)

# %%
