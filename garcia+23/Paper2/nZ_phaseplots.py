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
    r_sf = 500  # radii for sf in pc
    zsun = 0.02
    n_crit = 5e4
    cold_phase_t = (0, 100)
    warm_phase_t = (100, 5e4)
    hot_phase = (5.001e4, 1e9)
    temp_cuts = [cold_phase_t, warm_phase_t, hot_phase]
    tlabels = [
        r"CNM ($T < 100$ K)",
        r"WNM  ($ 100 < T \leq 5 \times 10^4$ K)",
        r"Hot ($T > 5 \times 10^4$ K)",
    ]
    lims = {
        ("gas", "density"): ((5e-30, "g/cm**3"), (1e-18, "g/cm**3")),
        ("ramses", "Metallicity"): (1e-6 * zsun, 5 * zsun),
        ("gas", "mass"): ((1e-2, "msun"), (1e6, "msun")),
    }

    if len(sys.argv) != 6:
        print(sys.argv[0], "usage:")
        print(
            "{} snapshot_dir start_snap end_snap step render_nickname".format(
                sys.argv[0]
            )
        )
        exit()
    else:
        print("********************************************************************")
        print(" rendering movie ")
        print("********************************************************************")

    datadir = sys.argv[1]
    logsfc_path = os.path.join(sys.argv[1], "logSFC")
    start_snapshot = int(sys.argv[2])
    end_snapshot = int(sys.argv[3])
    step = int(sys.argv[4])
    render_nickname = sys.argv[5]

    sim_run = os.path.basename(os.path.normpath(datadir))
    fpaths, snums = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )

    # first starburst in the CC-fid run
    # queiscent phase after 1st star burst, before 2nd snap 203 - 370
    # second starburst in the CC-fid run snap 377 - 389
    # after the second starburst snap 402 - 432

    # =============================================================================

    # datadir = os.path.expanduser("~/test_data/fid-broken-feedback/")
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
    # render_nickname = "test"

    # =============================================================================

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

    for i, sn in enumerate(fpaths):
        print(
            "# ________________________________________________________________________"
        )
        infofile = os.path.abspath(os.path.join(sn, f"info_{snums[i]}.txt"))
        print("# reading in", infofile)

        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        t_myr = float(ds.current_time.in_units("Myr"))
        redshift = ds.current_redshift

        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)

        ctr_at_code = np.array([x_center, y_center, z_center])

        mstar = np.sum(np.array(ad["star", "particle_mass"].to("Msun")))

        fig, ax = plt.subplots(1, 3, figsize=(10, 7), dpi=400, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0, wspace=0)

        axes = ax.ravel()

        for a, axis in enumerate(axes):
            print(temp_cuts[a])
            galaxy = ds.sphere(ctr_at_code, (r_sf, "pc"))
            galaxy_filtered = galaxy.include_inside(
                ("gas", "temperature"), temp_cuts[a][0], temp_cuts[a][1]
            )

            profile2d = galaxy_filtered.profile(
                # the x bin field, the y bin field
                [("gas", "density"), ("ramses", "Metallicity")],
                [("gas", "mass")],  # the profile field
                weight_field=None,  # sums each quantity in each bin
                n_bins=(200, 200),
                extrema=lims,
            )

            gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T
            print(gas_mass.shape)

            # metal = np.array(profile2d.y)
            # dens = np.array(profile2d.x)  # / 1.6e-24

            nt_image = axis.imshow(
                np.log10(gas_mass),
                origin="lower",
                extent=[
                    np.log10(lims[("gas", "density")][0][0] / m_h),
                    np.log10(lims[("gas", "density")][1][0] / m_h),
                    np.log10(lims[("ramses", "Metallicity")][0] / zsun),
                    np.log10(lims[("ramses", "Metallicity")][1] / zsun),
                ],
                cmap=cmr.tropical_r,
                vmin=np.log10(lims[("gas", "mass")][0][0]),
                vmax=np.log10(lims[("gas", "mass")][1][0]),
                aspect=1.6,
            )
            axis.set(xlabel=r"$\log\:n_{\rm H} \:{\rm (cm^{-3}})$")

            axis.text(
                0.97,
                0.08,
                r"$n_{ \rm crit}$",
                ha="right",
                va="bottom",
                color="k",
                rotation=90,
                transform=axis.transAxes,
            )
            axis.axvspan(np.log10(n_crit), np.log10(1e6), color="grey", alpha=0.5)

            axis.text(
                0.05,
                0.05,
                tlabels[a],
                ha="left",
                va="bottom",
                transform=axis.transAxes,
                fontsize=10,
            )

    ax[0].set(
        ylabel=r"$\log\:{\rm Metallicity\:(Z_{\odot})}$",
        xlabel=r"$\log\:n_{\rm H} \:{\rm (cm^{-3}})$",
        xlim=(
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
        ),
        ylim=(
            np.log10(lims[("ramses", "Metallicity")][0] / zsun),
            np.log10(lims[("ramses", "Metallicity")][1] / zsun),
        ),
    )
    ax[0].text(
        0.05,
        0.95,
        # ("{}" "\:" r"t = {:.2f} Myr" "\:" r"z = {:.2f} ").format(
        #     render_nickname,
        #     t_myr,
        #     redshift,
        # ),
        r"$t = {:.0f} $ Myr" "\n" r"$z =  {:.2f} $".format(t_myr, redshift),
        ha="left",
        va="top",
        color="black",
        transform=ax[0].transAxes,
        fontsize=10,
    )

    ax[0].xaxis.set_major_locator(plt.MaxNLocator(12))

    cbar_ax = ax[2].inset_axes([1.02, 0, 0.05, 1])
    bar = fig.colorbar(nt_image, cax=cbar_ax, pad=0)
    # bar .ax.xaxis.set_tick_params(pad=2)
    bar.set_label(r"$\mathrm{\log\:Total\:Mass\:(M_{\odot}})$")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.xaxis.set_ticks_position("top")

    output_path = os.path.join(
        render_container, "{}-{}.png".format(render_nickname, snums[i])
    )

    print("Saved", output_path)
    plt.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        # pad_inches=0.00,
    )
    # plt.show()
