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
from tools.ram_fields import ram_fields
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import h5py as h5
from scipy import interpolate

cell_fields, epf = ram_fields()


def _my_radial_velocity(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity").in_units("km/s")
    else:
        bv = data.ds.arr(np.zeros(3), "km/s")

    xv = data["gas", "velocity_x"] - bv[0]
    yv = data["gas", "velocity_y"] - bv[1]
    zv = data["gas", "velocity_z"] - bv[2]
    # what is supplied as center to the plotting routine
    center = data.get_field_parameter("center")
    x_hat = data["gas", "x"] - center[0]
    y_hat = data["gas", "y"] - center[1]
    z_hat = data["gas", "z"] - center[2]
    r = np.sqrt(x_hat * x_hat + y_hat * y_hat + z_hat * z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return xv * x_hat + yv * y_hat + zv * z_hat


yt.add_field(
    ("gas", "my_radial_velocity"),
    function=_my_radial_velocity,
    sampling_type="cell",
    units="km/s",
    take_log=False,
    validators=[
        ValidateParameter(["center", "bulk_velocity"]),
    ],
)


def read_sne_rate(logsfc_path, interuped=False):
    log_sfc = np.loadtxt(logsfc_path)
    if interuped is True:
        log_sfc_1 = np.loadtxt(logsfc_path + "-earlier")
        log_sfc = np.concatenate((log_sfc_1, log_sfc), axis=0)

    redshift = log_sfc[:, 2]
    sfr_binwidth_myr = 10

    t_myr = t_myr_from_z(redshift)
    # print(t_myr.min())
    running_total_sne = np.cumsum(np.arange(0, t_myr.size, 1))
    t_interp_points = np.arange(t_myr.min(), t_myr.max(), sfr_binwidth_myr)

    sne_interpolator = interpolate.interp1d(
        x=t_myr, y=running_total_sne, kind="previous"
    )
    sne_number = sne_interpolator(t_interp_points)
    # calculate the sfr in msun / yr
    sne_rate = np.gradient(sne_number) / (sfr_binwidth_myr * 1e6)

    # SNe properties
    # m_ejecta = log_sfc[:, 6]
    # e_thermal_injected = log_sfc[:, 7]
    # ejecta_zsun = log_sfc[:, 8]
    # let's do the accumulation of metals produced
    # mass_in_metals = m_ejecta * ejecta_zsun
    # total_mass_in_metals = np.cumsum(mass_in_metals)
    # position = log_sfc[:, 12:16]
    # np.linspace(0, len)
    return t_interp_points, sne_rate


f7_t, f7_rate = read_sne_rate(
    os.path.expanduser("~/container_tiramisu/sim_log_files/fs07_refine/logSN")
)
f3_t, f3_rate = read_sne_rate(
    os.path.expanduser("~/container_tiramisu/sim_log_files/fs035_ms10/logSN")
)
cc_t, cc_rate = read_sne_rate(
    os.path.expanduser("~/container_tiramisu/sim_log_files/CC-Fiducial/logSN"),
    interuped=True,
)
fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
ax.plot(cc_t, cc_rate)
ax.plot(f7_t, f7_rate)
ax.plot(f3_t, f3_rate)
plt.show()
# %%

processor_number = 0
fpaths, snums = filter_snapshots(
    os.path.expanduser("~/test_data/CC-Fiducial/"),
    380,
    412,
    sampling=1,
    str_snaps=True,
    snapshot_type="ramses_snapshot",
)
datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
sim_run = os.path.basename(os.path.normpath(datadir))

postprocessed_container = os.path.join(
    "..",
    "..",
    "..",
    "container_tiramisu",
    "post_processed",
    "gas_properties",
    sim_run,
)
dm_container = os.path.join(
    "..", "..", "..", "container_tiramisu", "post_processed", "dm_hop", sim_run
)
check_path(dm_container)
check_path(postprocessed_container)
phase_plot_per_snap = []
times = []
for i, sn in enumerate(fpaths):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snums[i]}.txt"))
    print("# reading in", infofile)
    try:
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()
        times.append(ds.current_time.in_units("Myr").value)
    except:
        print("having trouble reading snapshot, skipping")
        continue
    hop_catalogue = "{}/info_{}/info_{}.{}.h5".format(
        dm_container,
        snums[i],
        snums[i],
        processor_number,
    )

    if os.path.isfile(hop_catalogue) is True:
        print(">> catalogue already exists")
        print(hop_catalogue)
    else:
        hc = HaloCatalog(
            data_ds=ds,
            finder_method="hop",
            finder_kwargs={"ptype": "DM", "dm_only": False},
            output_dir=dm_container,
        )
        hc.create()

    cata_yt = h5.File(hop_catalogue, "r")

    dm_halo_m = np.max(np.array(cata_yt["particle_mass"][:]))  # Msun
    haloidx = np.argmax(dm_halo_m)  # most massive halo is the central halo

    # kpc is the default
    vir_rad = np.array(ds.arr(cata_yt["virial_radius"][:], "kpc").to("pc"))[haloidx]
    x_pos = ad["star", "particle_position_x"]
    y_pos = ad["star", "particle_position_y"]
    z_pos = ad["star", "particle_position_z"]
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)

    cata_yt.close()
    galaxy_center = ds.arr(
        np.array([x_center, y_center, z_center]), "code_length"
    ).in_units("pc")
    galaxy_radius = 500  # pc
    mass_hyrogen = 1.6735e-24  # grams
    end_region = 1e4  # "pc"

    try:
        vir_region = ds.sphere(
            galaxy_center, (vir_rad, "pc")
        )  # vir radius, usuall ~2kpc
        shell_thicknes = ds.arr(50, "pc")
        # SF region + 50 pc to calculate the quantities
        spherical_shell = vir_region.exclude_outside(
            ("index", "radius"),
            galaxy_radius,
            galaxy_radius + shell_thicknes.value,
            # vir_rad * 0.20,
            # vir_rad * 0.25,
            units="pc",
        )
        vrads = spherical_shell["radial_velocity"].to("km/s")
        outflows = spherical_shell.include_above(("gas", "my_radial_velocity"), 0)
        lims = {
            ("gas", "temperature"): ((1.5, "K"), (1e9, "K")),
            ("gas", "mass"): ((2e-2, "msun"), (2e3, "msun")),
            ("gas", "radial_velocity"): ((1, "km/s"), (1e4, "km/s")),
        }
        profile2d = outflows.profile(
            # the x bin field, the y bin field
            [("gas", "radial_velocity"), ("gas", "temperature")],
            fields=[("gas", "mass")],  # the profile field
            weight_field=None,  # sums each quantity in each bin
            n_bins=(100, 100),
            extrema=lims,
            # logs={("gas", "radial_velocity"): False},
        )
    except:
        print("can't do calculation, skipping")
        continue
    gas_mass_per_bin = np.array(profile2d["gas", "mass"].to("msun")).T
    phase_plot_per_snap.append(gas_mass_per_bin)


# %%
fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

time_avg = np.mean(phase_plot_per_snap, axis=0)

vt_image = ax.imshow(
    np.log10(time_avg),
    origin="lower",
    extent=[
        np.log10(lims[("gas", "radial_velocity")][0][0]),
        np.log10(lims[("gas", "radial_velocity")][1][0]),
        np.log10(lims[("gas", "temperature")][0][0]),
        np.log10(lims[("gas", "temperature")][1][0]),
    ],
    cmap="magma_r",
    vmin=np.log10(lims[("gas", "mass")][0][0]),
    vmax=np.log10(lims[("gas", "mass")][1][0]),
    aspect="auto",
)
ax.axhline(2, lw=1, ls="--", color="grey", alpha=0.8)
ax.axhline(np.log10(5e4), lw=1, ls="--", color="grey", alpha=0.8)


ax.text(
    0.95,
    0.12,
    r"cold",
    ha="right",
    va="bottom",
    color="grey",
    alpha=0.9,
    transform=ax.transAxes,
)

ax.text(
    0.95,
    0.35,
    r"warm",
    ha="right",
    va="bottom",
    color="grey",
    alpha=0.9,
    transform=ax.transAxes,
)

ax.text(
    0.95,
    0.55,
    r"hot",
    ha="right",
    va="bottom",
    color="grey",
    alpha=0.9,
    transform=ax.transAxes,
)


ax.text(
    0.05,
    0.95,
    r"VSFE",
    ha="left",
    va="top",
    transform=ax.transAxes,
)

ax.text(
    0.05,
    0.1,
    r"$t$ = {:.0f} - {:.0f} Myr".format(np.min(times), np.max(times)),
    ha="left",
    va="top",
    transform=ax.transAxes,
)


ax.set(xlabel=r"log $v_{\rm rad} \: [\rm km \, s^{-1}]$", ylabel=r"log Temperature [K]")

cbar_ax = ax.inset_axes([1.02, 0, 0.035, 1])
bar = fig.colorbar(vt_image, cax=cbar_ax, pad=0)
bar.set_label(r"$\mathrm{\log\:Gas\:Mass\: \left[M_{\odot} \right]}$")
bar.ax.xaxis.set_label_position("top")
bar.ax.xaxis.set_ticks_position("top")
cbar_ax.minorticks_on()


plt.show()
