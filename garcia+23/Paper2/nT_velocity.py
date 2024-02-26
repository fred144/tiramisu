"""
Metallicity as a function of density, with mass bins
Divided into three different gas phases
Region centered on star formation, radius of 0.5 kpc
"""

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

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

cell_fields, epf = ram_fields()


if __name__ == "__main__":
    m_h = 1.6735e-24  # grams
    r_sf = 1000  # radii for sf in pc
    zsun = 0.02
    n_crit = 5e4
    cold_phase_t = (0, 100)
    warm_phase_t = (100, 5e4)
    hot_phase = (5.001e4, 1e9)
    temp_cuts = [cold_phase_t, warm_phase_t, hot_phase]
    # tlabels = [
    #     r"CNM ($T < 100$ K)",
    #     r"WNM  ($ 100 < T \leq 5 \times 10^4$ K)",
    #     r"Hot ($T > 5 \times 10^4$ K)",
    # ]

    lims = {
        ("gas", "radial_velocity"): ((-3e3, "km/s"), (3e3, "km/s")),
        ("gas", "temperature"): ((50, "K"), (5e8, "K")),
        ("gas", "mass"): ((1e-2, "msun"), (1e6, "msun")),
    }
    # datadir = os.path.expanduser(
    #     "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
    # )
    datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    logsfc_path = os.path.expanduser(os.path.join(datadir, "logSFC"))

    fpaths, snums = filter_snapshots(
        datadir,
        304,
        304,
        sampling=1,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )
    # for the other rows
    fpaths1, snums1 = filter_snapshots(
        datadir,
        370,
        370,
        sampling=1,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

    fpaths2, snums2 = filter_snapshots(
        datadir,
        397,
        397,
        sampling=1,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

    # =============================================================================

    # datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    # logsfc_path = os.path.expanduser(
    #     "~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    # )
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     304,
    #     304,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # # for the other rows
    # fpaths1, snums1 = filter_snapshots(
    #     datadir,
    #     375,
    #     388,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    # fpaths2, snums2 = filter_snapshots(
    #     datadir,
    #     390,
    #     390,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    # fpaths3, snums3 = filter_snapshots(
    #     datadir,
    #     397,
    #     397,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    # =============================================================================
    render_nickname = "science_plots"
    # run save
    sim_run = os.path.basename(os.path.normpath(datadir))
    render_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "plots",
        sim_run,
        render_nickname,
    )
    check_path(render_container)

    snapshot_list = [
        [fpaths, snums],
        [fpaths1, snums1],
        [fpaths2, snums2],
    ]

    sfevariable = []
    sfe70 = []
    sfe35 = []

    myrs = []
    redshifts = []

    for sg, sn_group in enumerate(snapshot_list):
        # within each row  or grouping, read and update
        for r, (fpaths, snums) in enumerate(zip(sn_group[0], sn_group[1])):
            len_ofgroup = len(sn_group[0])
            print("# _________________________________________________________________")
            infofile = os.path.abspath(os.path.join(fpaths, f"info_{snums}.txt"))
            print("# reading in", infofile)

            ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
            ad = ds.all_data()

            t_myr = float(ds.current_time.in_units("Myr"))
            redshift = ds.current_redshift
            myrs.append(t_myr)
            redshifts.append(redshift)

            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)

            ctr_at_code = ds.arr([x_center, y_center, z_center], "code_length")

            mstar = np.sum(np.array(ad["star", "particle_mass"].to("Msun")))

            # for each grouping, go throught the CNM, WNM, and HOT phases

            sf_region = ds.sphere(ctr_at_code, (r_sf, "pc"))
            bulk_vel = sf_region.quantities.bulk_velocity()

            # Get the second sphere
            sf_region_corrected = ds.sphere(ctr_at_code, (r_sf, "pc"))

            # Set the bulk velocity field parameter
            sf_region_corrected.set_field_parameter("bulk_velocity", bulk_vel)

            profile2d = sf_region_corrected.profile(
                # the x bin field, the y bin field
                # metallicity is
                [("gas", "radial_velocity"), ("gas", "temperature")],
                [("gas", "mass")],  # the profile field
                weight_field=None,  # sums each quantity in each bin
                n_bins=(200, 200),
                extrema=lims,
                logs={("gas", "radial_velocity"): False},
            )

            gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T

            if sg == 0:
                sfevariable.append(gas_mass)
            elif sg == 1:
                sfe70.append(gas_mass)
            else:
                sfe35.append(gas_mass)

    # %%
    time_avg_vals = [
        np.mean(sfevariable, axis=0),
        np.mean(sfe70, axis=0),
        np.mean(sfe35, axis=0),
    ]

    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(5, 9),
        dpi=300,
    )

    plt.subplots_adjust(hspace=0, wspace=0)

    ax = ax.ravel()
    for a, phase_plot in enumerate(time_avg_vals):
        nz_image = ax[a].imshow(
            np.log10(phase_plot),
            origin="lower",
            extent=[
                lims[("gas", "radial_velocity")][0][0],
                lims[("gas", "radial_velocity")][1][0],
                np.log10(lims[("gas", "temperature")][0][0]),
                np.log10(lims[("gas", "temperature")][1][0]),
            ],
            cmap=cmr.torch_r,
            vmin=np.log10(lims[("gas", "mass")][0][0]),
            vmax=np.log10(lims[("gas", "mass")][1][0]),
            aspect=600,
        )
        row_time = (r"$t = {:.0f} $ Myr" "\n" r"$z =  {:.1f} $").format(
            myrs[a], redshifts[a]
        )

        ax[a].text(
            0.05,
            0.95,
            row_time,
            ha="left",
            va="top",
            color="black",
            transform=ax[a].transAxes,
            fontsize=10,
        )

    cbar_ax = ax[0].inset_axes([0, 1.02, 1, 0.05])
    bar = fig.colorbar(nz_image, cax=cbar_ax, pad=0, orientation="horizontal")
    # bar .ax.xaxis.set_tick_params(pad=2)
    bar.set_label(r"$\mathrm{\log\:Total\:Cell\:Mass\:\left[M_{\odot}\right]}$")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.xaxis.set_ticks_position("top")

    ax[1].set(ylabel="Temperature [K]")
    ax[2].set(xlabel=r"Radial Velocity $[ \mathrm{km \: s^{-1} }  ]$")
    plt.show()
    #     # %%
    #     for a, phase_plot in enumerate(time_avg_vals):
    #         nz_image = ax[sg, a].imshow(
    #             np.log10(phase_plot),
    #             origin="lower",
    #             extent=[
    #                 np.log10(lims[("gas", "density")][0][0] / m_h),
    #                 np.log10(lims[("gas", "density")][1][0] / m_h),
    #                 np.log10(lims[("ramses", "Metallicity")][0] / zsun),
    #                 np.log10(lims[("ramses", "Metallicity")][1] / zsun),
    #             ],
    #             cmap=cmr.tropical_r,
    #             vmin=np.log10(lims[("gas", "mass")][0][0]),
    #             vmax=np.log10(lims[("gas", "mass")][1][0]),
    #             aspect=1.6,
    #         )

    #         # ax[sg, a].set(xlabel=r"$\log\:n_{\rm H} \:{\rm (cm^{-3}})$")

    #         ax[sg, a].text(
    #             0.97,
    #             0.08,
    #             r"$n_{ \rm crit}$",
    #             ha="right",
    #             va="bottom",
    #             color="k",
    #             rotation=90,
    #             transform=ax[sg, a].transAxes,
    #         )
    #         ax[sg, a].axvspan(np.log10(n_crit), np.log10(1e6), color="grey", alpha=0.5)

    #         if sg == 0:
    #             ax[sg, a].text(
    #                 0.05,
    #                 0.95,
    #                 tlabels[a],
    #                 ha="left",
    #                 va="top",
    #                 transform=ax[sg, a].transAxes,
    #                 fontsize=10,
    #             )

    #     if len_ofgroup == 1:
    #         row_time = (r"$t = {:.0f} $ Myr" "\n" r"$z =  {:.1f} $").format(
    #             myrs[0],
    #             redshifts[0],
    #         )

    #     else:
    #         row_time = (
    #             r"$t = {:.0f} - {:.0f} $ Myr" "\n" r"$z =  {:.1f} - {:.1f} $"
    #         ).format(
    #             np.array(myrs).min(),
    #             np.array(myrs).max(),
    #             np.array(redshifts).max(),
    #             np.array(redshifts).min(),
    #         )

    #     ax[sg, 0].text(
    #         0.05,
    #         0.05,
    #         row_time,
    #         ha="left",
    #         va="bottom",
    #         color="black",
    #         transform=ax[sg, 0].transAxes,
    #         fontsize=10,
    #     )

    # ax[sg, a].set(
    #     xlim=(
    #         np.log10(lims[("gas", "density")][0][0] / m_h),
    #         np.log10(lims[("gas", "density")][1][0] / m_h),
    #     ),
    #     ylim=(-5.5, 0.5),
    # )
    # ax[sg, a].xaxis.set_major_locator(plt.MaxNLocator(12))
    # ax[sg, a].yaxis.set_major_locator(plt.MaxNLocator(6))

    # cbar_ax = ax[0, 0].inset_axes([0, 1.02, 3, 0.05])
    # bar = fig.colorbar(nz_image, cax=cbar_ax, pad=0, orientation="horizontal")
    # # bar .ax.xaxis.set_tick_params(pad=2)
    # bar.set_label(r"$\mathrm{\log\:Total\:Cell\:Mass\:\left[M_{\odot}}\right]$")
    # bar.ax.xaxis.set_label_position("top")
    # bar.ax.xaxis.set_ticks_position("top")
    # # cbar_ax.xaxis.set_major_locator(plt.MaxNLocator(8))

    # fig.text(0.5, 0.13, r"$\log\:n_{\rm H} \:{\rm \left[cm^{-3}} \right]$", ha="center")

    # fig.text(
    #     0.06,
    #     0.5,
    #     r"$\log\:{\rm Metallicity\:\left[Z_{\odot}\right]}$",
    #     va="center",
    #     rotation="vertical",
    # )

    # output_path = os.path.join(render_container, "multiphase_nZ.png")

    # print("Saved", output_path)
    # plt.savefig(
    #     output_path,
    #     dpi=300,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )

    # plt.show()
