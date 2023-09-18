import sys

sys.path.append("../")
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import colors

import yt

import matplotlib
import matplotlib.patheffects as patheffects
from scipy.spatial.transform import Rotation as R
from yt.visualization.volume_rendering.api import Scene
from scipy.ndimage import gaussian_filter
from tools.check_path import check_path
from tools import plotstyle
from tools.fscanner import filter_snapshots
from tools.ram_fields import ram_fields


def draw_frame(
    gas_array, luminosity, ax, fig, wdth, t_myr, redshift, label, star_bins=2000
):
    lum_range = (3e33, 3e36)
    pxl_size = (wdth / star_bins) ** 2
    lum_alpha = 1
    gas_alpha = 0.55
    surface_brightness = luminosity / pxl_size
    # clean up edges
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position("none")
    ax.axis("off")
    # luminosity
    lum = ax.imshow(
        surface_brightness,
        cmap="inferno",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
        alpha=lum_alpha,
    )

    # three panels gas density
    gas = ax.imshow(
        gas_array,
        cmap="cubehelix",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(0.008, 0.32),
        alpha=gas_alpha,
    )

    # add scale
    scale = patches.Rectangle(
        xy=(wdth / 2 * 0.35, -wdth / 2 * 0.85),
        width=wdth / 2 * 0.5,
        height=0.010 * wdth / 2,
        linewidth=0,
        edgecolor="white",
        facecolor="white",
        clip_on=False,
        alpha=0.8,
    )
    ax.text(
        wdth / 2 * 0.61,
        -wdth / 2 * 0.90,
        r"$\mathrm{{{:.0f}\:pc}}$".format(wdth / 2 * 0.5),
        ha="center",
        va="center",
        color="white",
        alpha=0.8,
        # fontproperties=leg_font,
        # fontsize=14
    )
    ax.add_patch(scale)

    # add the luminosity color bar
    lum_cbar_ax = ax.inset_axes([0.10, 0.03, 0.010, 0.18])
    lum_cbar = fig.colorbar(lum, cax=lum_cbar_ax, pad=0)
    lum_cbar_ax.set_ylabel(r"${\rm erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}}$", fontsize=10)
    # add the gas color bar
    gas_cbar_ax = ax.inset_axes([0.08, 0.03, 0.010, 0.18])
    gas_cbar = fig.colorbar(gas, cax=gas_cbar_ax, pad=0)
    gas_cbar_ax.yaxis.set_ticks_position("left")
    gas_cbar_ax.set_ylabel(r"$\mathrm{ g \: cm^{-2}}$", fontsize=10, labelpad=-16)

    # add time and redshift
    ax.text(
        0.05,
        0.96,
        (
            "{}" "\n" r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$"
        ).format(
            label,
            t_myr,
            redshift,
        ),
        ha="left",
        va="top",
        color="white",
        transform=ax.transAxes,
    )


# %%
if __name__ == "__main__":
    # yt.enable_parallelism()
    plt.style.use("dark_background")

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
    start_snapshot = int(sys.argv[2])
    end_snapshot = int(sys.argv[3])
    step = int(sys.argv[4])
    render_nickname = sys.argv[5]

    ## local path for test
    # datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")
    # datadir = os.path.relpath("../../garcia23_testdata/fs07_refine")
    # start_snapshot = 500
    # end_snapshot = 500
    # step = 1
    # render_nickname = "gas_and_lum"

    snaps, snap_strings = filter_snapshots(
        datadir, start_snapshot, end_snapshot, sampling=step, str_snaps=True
    )
    sim_run = os.path.basename(os.path.normpath(datadir))
    render_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "renders",
        render_nickname,  #!!! replace with render path name
        sim_run,
    )
    check_path(render_container)
    pop2_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "pop2", sim_run
    )
    cell_fields, epf = ram_fields()

    # timelapse paramaters
    static_plt_wdth = 480
    zoom_plt_wdth = 120
    star_bins = 2000
    pxl_size = (static_plt_wdth / star_bins) ** 2  # pc

    pan_frames = 400  # number of frames to use up for each rotation,
    num_rots = 4  # number of 360 degreee rotations
    rotation_interval = np.linspace(0, 2 * num_rots, pan_frames, endpoint=False) * np.pi
    zoom_interval = np.concatenate(
        [
            static_plt_wdth * np.ones(int(pan_frames / 4)),
            np.linspace(static_plt_wdth, zoom_plt_wdth, int(pan_frames / 4)),
            zoom_plt_wdth * np.ones(int(pan_frames / 4)),
            np.linspace(zoom_plt_wdth, static_plt_wdth, int(pan_frames / 4)),
        ]
    )
    pause_and_rotate = [304, 390]

    for i, sn in enumerate(snaps):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)

        pop2_path = glob.glob(
            os.path.join(pop2_container, "pop2-{}-*".format(snap_strings[i]))
        )[0]
        if os.path.isfile(pop2_path) is True:
            print("# file already exists")
            print("# reading in", pop2_path)
            pop2_data = np.loadtxt(pop2_path)
            t_myr = pop2_data[0, 0]
            redshift = pop2_data[1, 0]
            ctr_at_code = pop2_data[2:5, 0]
            pop2_log_lums = pop2_data[:, 3]
            pop2_xyz = pop2_data[:, 4:7]  # pc
            # pop2_y = pop2_data[:, 5]
            # pop2_z = pop2_data[:, 6]

        else:
            print("*** no post-processed pop2")
            continue

        if int(snap_strings[i]) in pause_and_rotate:
            # reset the star positions every loop
            print(">>> Rotating View")

            rr = 0  # rotation sequence restart from

            for rot_i, (rotation_angle, plt_wdth) in enumerate(
                zip(rotation_interval[rr:], zoom_interval[rr:]), start=rr
            ):
                # along (x,y,z) axis
                r = R.from_rotvec(rotation_angle * np.array([0, 1, 0]))
                rotation_matrix = r.as_matrix()
                print(">> rotation angle", rotation_angle, "of", rotation_interval[-1])
                print(">> frame", rot_i, "of", pan_frames)
                # rotate stars
                rotated_star_positions = np.dot(pop2_xyz, rotation_matrix)
                lums, _, _ = np.histogram2d(
                    rotated_star_positions[:, 0],
                    rotated_star_positions[:, 1],
                    bins=star_bins,
                    weights=10**pop2_log_lums,
                    normed=False,
                    range=[
                        [-plt_wdth / 2, plt_wdth / 2],
                        [-plt_wdth / 2, plt_wdth / 2],
                    ],
                )
                lums = lums.T

                gas = yt.OffAxisProjectionPlot(
                    ds,
                    normal=np.dot(np.array([0.0, 0.0, 1.0]), rotation_matrix),
                    fields=("gas", "density"),
                    center=ctr_at_code,
                    north_vector=np.array([0.0, 1.0, 0.0]),
                    width=(plt_wdth, "pc"),
                    # resolution=2000,
                )

                gas_frb = gas.to_fits_data()
                gas_array = np.array(gas_frb[0].data)  # .T

                fig, ax = plt.subplots(
                    figsize=(13, 13),
                    dpi=300,
                    facecolor=cm.Greys_r(0),
                )
                draw_frame(
                    gas_array, lums, ax, fig, plt_wdth, t_myr, redshift, label=sim_run
                )
                output_path = os.path.join(
                    render_container,
                    "{}-{}_rotated-{}.png".format(
                        render_nickname, snap_strings[i], str(rot_i).zfill(3)
                    ),
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
                print(">Saved:", output_path)
                plt.close()
        else:
            plt_wdth = static_plt_wdth
            lums, _, _ = np.histogram2d(
                pop2_xyz[:, 0],
                pop2_xyz[:, 1],
                bins=star_bins,
                weights=10**pop2_log_lums,
                range=[
                    [-plt_wdth / 2, plt_wdth / 2],
                    [-plt_wdth / 2, plt_wdth / 2],
                ],
            )
            lums = lums.T

            gas = yt.ProjectionPlot(
                ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=ctr_at_code
            )
            gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
            gas_array = np.array(gas_frb["gas", "density"])
            # %%
            fig, ax = plt.subplots(figsize=(13, 13), dpi=400, facecolor=cm.Greys_r(0))
            draw_frame(
                gas_array, lums, ax, fig, plt_wdth, t_myr, redshift, label=sim_run
            )
            output_path = os.path.join(
                render_container, "{}-{}.png".format(render_nickname, snap_strings[i])
            )
            # plt.show()
            plt.savefig(
                os.path.expanduser(output_path),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )

            print(">>> saved:", output_path)
            # plt.close()
