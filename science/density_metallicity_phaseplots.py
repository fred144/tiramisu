import sys

sys.path.append("../")

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

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0

cell_fields, epf = ram_fields()


if __name__ == "__main__":
    m_h = 1.6735e-24  # grams
    r_sf = 500  # radii for sf in pc
    zsun = 0.02
    n_crit = 5e4
    mc_t = (1, 90)
    wnm_t = (101, 9e5)
    ion_t = (1e6, 1e8)
    temp_cuts = [mc_t, wnm_t, ion_t]
    lims = {
        ("gas", "density"): ((5e-31, "g/cm**3"), (1e-18, "g/cm**3")),
        ("ramses", "Metallicity"): (5e-8 * zsun, 10 * zsun),
        ("gas", "mass"): ((1e-2, "msun"), (1e6, "msun")),
    }

    if len(sys.argv) != 7:
        print(sys.argv[0], "usage:")
        print(
            "{} snapshot_dir logSFC start_snap end_snap step render_nickname".format(
                sys.argv[0]
            )
        )
        exit()
    else:
        print("********************************************************************")
        print(" rendering movie ")
        print("********************************************************************")

    datadir = sys.argv[1]
    logsfc_path = sys.argv[2]
    start_snapshot = int(sys.argv[3])
    end_snapshot = int(sys.argv[4])
    step = int(sys.argv[5])
    render_nickname = sys.argv[6]

    sim_run = os.path.basename(os.path.normpath(datadir))
    fpaths, snums = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )

    # =============================================================================
    # datadir = os.path.expanduser("~/test_data/fid-broken-feedback/")
    # logsfc_path = os.path.expanduser(
    #     "~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    # )
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     304,
    #     390,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # render_nickname = "gas_metallicity"

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

    m_vir = []
    r_vir = []
    tot_m_star = []
    t = []
    z = []
    snap = []

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
        # tot_m_star.append(mstar)

        # ad_filt = ad.include_inside(

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
        fig, ax = plt.subplots(1, 3, figsize=(13, 5), dpi=300, sharex=True, sharey=True)
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
                n_bins=(125, 125),
                extrema=lims,
            )

            gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T
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
                cmap="rainbow",
                vmin=np.log10(lims[("gas", "mass")][0][0]),
                vmax=np.log10(1e6),
                aspect=1.6,
            )
            axis.set(xlabel=r"$\log_{10}\:n_{\rm H} \:{\rm (cm^{-3}})$")

            axis.text(
                0.97,
                0.08,
                r"$n_{ \rm crit}$",
                ha="right",
                va="bottom",
                color="white",
                rotation=90,
                transform=axis.transAxes,
            )
            axis.axvspan(np.log10(n_crit), np.log10(1e6), color="grey")

            axis.text(
                0.03,
                0.08,
                r"$ {:.0e} K < T < {:.0e} K $".format(temp_cuts[a][0], temp_cuts[a][1]),
                ha="left",
                va="bottom",
                transform=axis.transAxes,
            )

        ax[0].set(
            ylabel=r"$\log_{10}\:{\rm Metallicity\:(Z_{\odot})}$",
            xlabel=r"$\log_{10}\:n_{\rm H} \:{\rm (cm^{-3}})$",
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
            0.01,
            1.1,
            ("{}" "\:" r"t = {:.2f} Myr" "\:" r"z = {:.2f} ").format(
                render_nickname,
                t_myr,
                redshift,
            ),
            ha="left",
            va="top",
            color="black",
            transform=ax[0].transAxes,
        )

        ax[0].xaxis.set_major_locator(plt.MaxNLocator(12))

        cbar_ax = ax[0].inset_axes([1, 1.02, 2, 0.05])
        bar = fig.colorbar(nt_image, cax=cbar_ax, pad=0, orientation="horizontal")
        # bar .ax.xaxis.set_tick_params(pad=2)
        bar.set_label(r"$\mathrm{\log_{10}\:Total\:Mass\:(M_{\odot}})$")
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
        plt.close()
