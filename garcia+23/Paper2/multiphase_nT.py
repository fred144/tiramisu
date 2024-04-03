import sys

sys.path.append("../../")

import yt
import numpy as np
import os
from tools.fscanner import filter_snapshots
from tools.ram_fields import ram_fields
import h5py as h5
from yt.funcs import mylog
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import warnings
import matplotlib.pyplot as plt

import matplotlib.patheffects as patheffects
from scipy.spatial.transform import Rotation as R
from yt.visualization.volume_rendering.api import Scene
from scipy.ndimage import gaussian_filter
from tools.check_path import check_path
from tools import plotstyle
from tools.fscanner import filter_snapshots
from tools.ram_fields import ram_fields
import cmasher as cmr
import matplotlib.colors as colors

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0


# datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")
# datadir = os.path.relpath("../../garcia23_testdata/fs07_refine")
# start_snapshot = 500
# end_snapshot = 500
# step = 1

cell_fields, epf = ram_fields()
# datadir = os.path.relpath("../../cosm_test_data/refine")

# datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-Fiducial")


# datadir = "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"


snaps_cc, snap_strings_cc = filter_snapshots(
    os.path.expanduser("~/test_data/CC-Fiducial/"), 397, 397, sampling=1, str_snaps=True
)
snaps_70, snap_strings_70 = filter_snapshots(
    os.path.expanduser("~/test_data/fs07_refine/"),
    1366,
    1366,
    sampling=1,
    str_snaps=True,
)

snaps_35, snap_strings_35 = filter_snapshots(
    os.path.expanduser("~/test_data/fs035_ms10/"),
    384,
    384,
    sampling=1,
    str_snaps=True,
)


# simulation_run = datadir
plot_name = "CC-Fid_nvsT_sfregion"
zsun = 0.02
r_sf = 500
lims = {
    ("gas", "density"): ((5e-31, "g/cm**3"), (1e-18, "g/cm**3")),
    ("gas", "temperature"): ((1.5, "K"), (1e9, "K")),
    ("gas", "mass"): ((2e-6, "msun"), (2e6, "msun")),
    ("gas", "radial_velocity"): ((-1e6, "km/s"), (1e6, "km/s")),
    ("ramses", "Metallicity"): (1e-6, 2),
}
m_h = 1.6735e-24  # grams


sim_run = "Multi"
dm_container = os.path.join("..", "..", "container_tiramisu", "dm_hop", sim_run)
if not os.path.exists(dm_container):
    print("Creating container", dm_container)
    os.makedirs(dm_container)
gas_container = os.path.join(
    "..", "..", "..", "container_tiramisu", "plots", plot_name, sim_run
)
if not os.path.exists(gas_container):
    print("Creating container", gas_container)
    os.makedirs(gas_container)


m_vir = []
r_vir = []
tot_m_star = []
t = []
z = []
snap = []


def fill_column(ax, ad, drawbars=False):
    mstar = np.sum(np.array(ad["star", "particle_mass"].to("Msun")))
    tot_m_star.append(mstar)
    # efficiency = mstar / dm_halo_m
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)

    ctr_at_code = ds.arr([x_center, y_center, z_center], "code_length")

    sf_region = ds.sphere(ctr_at_code, (r_sf, "pc"))

    bulk_vel = sf_region.quantities.bulk_velocity()

    # Get the second sphere
    sf_region_corrected = ds.sphere(ctr_at_code, (r_sf, "pc"))

    # Set the bulk velocity field parameter
    sf_region_corrected.set_field_parameter("bulk_velocity", bulk_vel)

    # useful info
    current_time = float(ds.current_time.in_units("Myr"))
    redshft = ds.current_redshift

    # plot = yt.PhasePlot(
    #     galaxy,
    #     ("gas", "density"),
    #     ("gas", "temperature"),
    #     ("gas", "mass"),
    #     weight_field=None,
    # )
    # Set the units of mass to be in solar masses (not the default in cgs)
    # plot.set_unit(("gas", "mass"), "Msun")
    # plot.save()

    profile2d = sf_region_corrected.profile(
        # the x bin field, the y bin field
        [("gas", "density"), ("gas", "temperature")],
        fields=[
            ("gas", "mass"),
        ],  # the profile field
        weight_field=None,  # sums each quantity in each bin
        n_bins=(125, 125),
        extrema=lims,
        logs={("gas", "radial_velocity"): False},
    )

    # average values
    ad["ramses", "Metallicity"] = (
        ad["ramses", "Metallicity"] * 50
    )  # / zsun  # turn into zsun

    # make a new field for converting to solar metals
    profile2d_average = sf_region_corrected.profile(
        # the x bin field, the y bin field
        [("gas", "density"), ("gas", "temperature")],
        fields=[
            ("gas", "radial_velocity"),
        ]("ramses", "Metallicity"),
        n_bins=(125, 125),
        weight_field=("gas", "mass"),
        extrema=lims,
        logs={("gas", "radial_velocity"): False},
    )

    gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T
    gas_velocity = np.array(profile2d_average["gas", "radial_velocity"].to("km/s")).T
    gas_metallicity = np.array(profile2d_average["ramses", "Metallicity"]).T

    temp = np.array(profile2d.y)
    dens = np.array(profile2d.x)

    nt_image = ax[0].imshow(
        np.log10(gas_mass),
        origin="lower",
        extent=[
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
            np.log10(lims[("gas", "temperature")][0][0]),
            np.log10(lims[("gas", "temperature")][1][0]),
        ],
        cmap="magma_r",
        vmin=np.log10(lims[("gas", "mass")][0][0]),
        vmax=np.log10(lims[("gas", "mass")][1][0]),
        aspect=1.3,
    )
    nt_image_v = ax[1].imshow(
        gas_velocity,
        origin="lower",
        extent=[
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
            np.log10(lims[("gas", "temperature")][0][0]),
            np.log10(lims[("gas", "temperature")][1][0]),
        ],
        cmap=cmr.viola,
        # vmin=0,
        # vmax=1e3,
        norm=colors.SymLogNorm(linthresh=100, linscale=1, vmin=-2e3, vmax=2e3),
        aspect=1.3,
    )

    nt_image_Z = ax[2].imshow(
        np.log10(gas_metallicity),
        origin="lower",
        extent=[
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
            np.log10(lims[("gas", "temperature")][0][0]),
            np.log10(lims[("gas", "temperature")][1][0]),
        ],
        cmap=cmr.savanna,
        vmax=np.log10(1),
        vmin=np.log10(3e-5),
        aspect=1.3,
    )

    # multiphase
    # get the critical SF region
    temp_arr = np.geomspace(
        lims[("gas", "temperature")][0][0], lims[("gas", "temperature")][1][0]
    )

    ncr = 4
    n_crit = 5e4 * (temp_arr / 100) * ((1 + redshft) / 10) ** 2 * (ncr / 4) ** -2
    for a in range(3):
        ax[a].axhline(2, lw=1, ls="--", color="grey", alpha=0.8)
        ax[a].axhline(np.log10(5e4), lw=1, ls="--", color="grey", alpha=0.8)

    # SNe
    ax[0].fill_between(
        np.array([-7, -1.5]), 6, 9, facecolor="grey", alpha=0.8, zorder=-1
    )

    ax[0].fill_between(
        np.log10(n_crit), np.log10(temp_arr), facecolor="grey", alpha=0.8, zorder=-1
    )

    ax[0].text(
        0.98,
        0.03,
        r"$n_{ \rm H, crit}$",
        ha="right",
        va="bottom",
        color="white",
        transform=ax[0].transAxes,
    )

    ax[0].text(
        0.05,
        0.85,
        r"SNe",
        ha="left",
        va="bottom",
        color="white",
        transform=ax[0].transAxes,
    )

    ax[0].text(
        0.05,
        0.12,
        r"cold",
        ha="left",
        va="bottom",
        color="grey",
        alpha=0.9,
        transform=ax[0].transAxes,
    )

    ax[0].text(
        0.05,
        0.25,
        r"warm",
        ha="left",
        va="bottom",
        color="grey",
        alpha=0.9,
        transform=ax[0].transAxes,
    )
    ax[0].text(
        0.05,
        0.55,
        r"hot",
        ha="left",
        va="bottom",
        color="grey",
        alpha=0.9,
        transform=ax[0].transAxes,
    )

    ax[0].text(
        0.65,
        0.95,
        (r"$\mathrm{{t = {:.0f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$").format(
            current_time, redshft
        ),
        ha="left",
        va="top",
        transform=ax[0].transAxes,
    )

    ax[2].minorticks_on()
    ax[2].xaxis.set_major_locator(plt.MaxNLocator(12))
    ax[2].yaxis.set_major_locator(plt.MaxNLocator(10))

    if drawbars is True:
        cbar_ax = ax[0].inset_axes([1.02, 0, 0.035, 1])
        bar = fig.colorbar(nt_image, cax=cbar_ax, pad=0)
        bar.set_label(r"$\mathrm{\log\:Gas\:Mass\: \left[M_{\odot} \right]}$")
        bar.ax.xaxis.set_label_position("top")
        bar.ax.xaxis.set_ticks_position("top")
        cbar_ax.minorticks_on()

        cbar_ax = ax[1].inset_axes([1.02, 0, 0.035, 1])
        bar = fig.colorbar(nt_image_v, cax=cbar_ax, pad=0)
        bar.set_label(r"$\mathrm{Radial\:Velocity\left[M_{\odot} \right]}$")
        bar.ax.xaxis.set_label_position("top")
        bar.ax.xaxis.set_ticks_position("top")
        cbar_ax.minorticks_on()

        cbar_ax = ax[2].inset_axes([1.02, 0, 0.035, 1])
        bar = fig.colorbar(nt_image_Z, cax=cbar_ax, pad=0)
        bar.set_label(
            r"$\mathrm{\log\:Mean\:Gas\:Metallicity\: \left[Z_{\odot} \right]}$"
        )
        bar.ax.xaxis.set_label_position("top")
        bar.ax.xaxis.set_ticks_position("top")
        cbar_ax.minorticks_on()

    return nt_image, nt_image_v, nt_image_Z


snaps = [
    (snaps_cc, snap_strings_cc),
    (snaps_70, snap_strings_70),
    (snaps_35, snap_strings_35),
]
# %%

fig, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=400, sharex=True, sharey=True)


for i, s in enumerate(snaps):
    for a, (sn, snap_strings) in enumerate(zip(s[0], s[1])):
        print("# _______________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings}.txt"))
        print("# reading in", infofile)

        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        # hop_catalogue = os.path.join(
        #     dm_container,
        #     f"info_{snap_strings[i]}",
        #     f"info_{snap_strings[i]}.{processor_number}.h5",
        # )

        # if os.path.isfile(hop_catalogue) is True:
        #     # print("file already exists")
        #     pass
        # else:
        #     hc = HaloCatalog(
        #         data_ds=ds,
        #         finder_method="hop",
        #         finder_kwargs={"ptype": "DM", "dm_only": False},
        #         output_dir=dm_container,
        #     )
        #     hc.create()
        # need to read in using yt for virial radius for some reason unknown units in catalogue
        # cata_yt = yt.load(hop_catalogue)
        # cata_yt = cata_yt.all_data()
        # dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
        # haloidx = np.argmax(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
        # vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]
        # m_vir.append(dm_halo_m)
        # r_vir.append(vir_rad)

        # center of the major halo
        # halo_x = ds.arr(cata_yt["all", "particle_position_x"]).to("pc")[haloidx]
        # halo_y = ds.arr(cata_yt["all", "particle_position_y"]).to("pc")[haloidx]
        # halo_z = ds.arr(cata_yt["all", "particle_position_z"]).to("pc")[haloidx]
        # galaxy = ds.sphere(
        #     [halo_x, halo_y, halo_z],
        #     ds.arr(cata_yt["all", "virial_radius"]).to("pc")[haloidx],
        # )
        # =============================================================================
        # / 1.6e-24

        current_time = float(ds.current_time.in_units("Myr"))
        redshft = ds.current_redshift
        t.append(current_time)
        z.append(redshft)
        if i == 2:
            nt_image, nt_image_v, nt_image_Z = fill_column(
                ax[:, i].ravel(), ad, drawbars=True
            )
        else:
            nt_image, nt_image_v, nt_image_Z = fill_column(ax[:, i].ravel(), ad)

        # ax.text(
        #     0.65,
        #     0.95,
        #     (
        #         r"$\mathrm{{t = {:.2f} \: Myr}}$"
        #         "\n"
        #         r"$\mathrm{{z = {:.2f} }}$"
        #         "\n"
        #         r"$\mathrm{{SFE_{{halo}} = {:.2e}}}$"
        #         "\n"
        #         r"${{\rm R_{{ SF \, region}}}} = 500 {{\rm pc}}$"
        #         "\n"
        #         r"${{\rm run-}}\mathrm{{{}}}$"
        #     ).format(current_time, redshft, efficiency, sim_run),
        #     ha="left",
        #     va="top",
        #     transform=ax.transAxes,
        # )

ax[1, 0].set(
    ylabel=r"$\log\:{\rm Temperature\: \left[K\right]}$",
)
ax[2, 1].set(
    xlabel=r"$\log\:n_{\rm H} \:{\rm \left[cm^{-3} \right] }$",
    xlim=(
        np.log10(lims[("gas", "density")][0][0] / m_h),
        np.log10(lims[("gas", "density")][1][0] / m_h),
    ),
    ylim=(
        np.log10(lims[("gas", "temperature")][0][0]),
        np.log10(lims[("gas", "temperature")][1][0]),
    ),
)

ax[0, 0].set(title=r"VSFE")
ax[0, 1].set(title=r"high SFE")
ax[0, 2].set(title=r"low SFE")

plt.subplots_adjust(hspace=-0.1, wspace=0)

save_path = os.path.join(
    gas_container,
    r"{}-{}.png".format(
        plot_name,
        snap_strings[i],
        "{:.2f}".format(current_time).replace(".", "_"),
    ),
)
print("Saved", save_path)
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/phaseplot_nT.png",
    # save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.show()
