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
from yt.funcs import mylog
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
import warnings
import cmasher as cmr
from src.lum.lum_lookup import lum_look_up_table
from src.lum.pop2 import get_star_ages
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

plt.rcParams.update(
    {
        # "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.size": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 7,
        "ytick.minor.size": 7,
    }
)
plt.style.use("dark_background")


def linear_cmap(start_alpha, end_alpha, module, cmap):
    ncolors = 256
    if module == "mpl":
        gas_cmap_arr = plt.get_cmap(cmap)(range(ncolors))
    elif module == "cmr":
        gas_cmap_arr = cmr.take_cmap_colors(cmap, ncolors)
        gas_cmap_arr = np.vstack(
            [np.array(gas_cmap_arr).T, np.zeros(np.shape(gas_cmap_arr)[0]).T]
        ).T
    # dictate alphas manually
    # decide which part of the cmap is increasing in alpha
    # the final transparencey is dictatted by

    gas_cmap_arr[0:150, -1] = np.linspace(
        start_alpha, end_alpha, gas_cmap_arr[0:150, -1].size
    )
    gas_cmap_arr[150:, -1] = np.ones(gas_cmap_arr[150:, -1].size) * end_alpha
    gascmap = LinearSegmentedColormap.from_list(
        name=cmap + "_custom", colors=gas_cmap_arr
    )
    return gascmap


def draw_frame(
    gas_array,
    temp_arry,
    stellar_mass_dens_arr,
    ax,
    fig,
    wdth,
    t_myr,
    redshift,
    label,
    star_bins=2000,
):
    stelllar_range = (20, 2e4)
    gas_range = (30, 2e4)
    temp_range = (6e3, 3e5)

    pxl_size = (wdth / star_bins) ** 2
    lum_alpha = 1

    surface_dens = stellar_mass_dens_arr / pxl_size
    print(np.max(surface_dens))
    print(np.max(gas_array))
    # clean up edges
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position("none")
    ax.axis("off")

    # dictate alphas manually
    # decide which part of the cmap is increasing in alpha
    # the final transparencey is dictatted by

    gascmap = linear_cmap(0.2, 0.80, "cmr", "cmr.dusk")
    # tempcmap = linear_cmap(0.40, 0.70, "cmr", "cmr.ember")
    lumcmap = "cmr.ember"
    gauss_sig = 1

    # making a meshgrid for contours
    x = np.linspace(-0.5 * wdth, 0.5 * wdth, gas_array.shape[0])
    y = np.linspace(-0.5 * wdth, 0.5 * wdth, gas_array.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = np.array(gas_array)

    # three panels gas density
    sdense = ax.imshow(
        surface_dens,
        cmap=lumcmap,
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=stelllar_range[0], vmax=stelllar_range[1]),
        alpha=lum_alpha,
    )
    gas = ax.imshow(
        gaussian_filter(gas_array, gauss_sig),
        cmap=gascmap,
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(gas_range[0], gas_range[1]),
    )

    lcmap = matplotlib.colormaps["Set2"]
    lcmap = lcmap(np.linspace(0, 1, 8))

    density_contours = ax.contour(
        X,
        Y,
        Z,
        [100.0, 200.0, 500.0, 1e3],
        colors=[lcmap[0], lcmap[1], lcmap[4], lcmap[5]],
        linewidths=1,
    )

    ax.text(
        0.98,
        0.95,
        r"Contours $\left[ \mathrm{ M_\odot \: pc^{-2} }  \right]$",
        ha="right",
        va="top",
        color="white",
        transform=ax.transAxes,
    )

    ax.text(
        0.95, 0.91, r"100", ha="right", va="top", color=lcmap[0], transform=ax.transAxes
    )

    ax.text(
        0.95, 0.88, r"200", ha="right", va="top", color=lcmap[1], transform=ax.transAxes
    )

    ax.text(
        0.95, 0.85, r"500", ha="right", va="top", color=lcmap[4], transform=ax.transAxes
    )

    ax.text(
        0.95,
        0.82,
        r"1000",
        ha="right",
        va="top",
        color=lcmap[5],
        transform=ax.transAxes,
    )

    # ax.clabel(density_contours, inline=False)

    # temp = ax.imshow(
    #     gaussian_filter(temp_array, gauss_sig),
    #     cmap=tempcmap,
    #     interpolation="gaussian",
    #     origin="lower",
    #     extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
    #     norm=LogNorm(temp_range[0], temp_range[1]),
    # )

    ##add scale
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

    ##add the luminosity color bar
    sdense_cbar_ax = ax.inset_axes([0.04, 0.03, 0.30, 0.015])
    sdense_cbar = fig.colorbar(
        sdense, cax=sdense_cbar_ax, pad=0.1, orientation="horizontal"
    )
    sdense_cbar_ax.set_title(r"$\Sigma_{\rm stars}$  [$\mathrm{M_\odot\:pc^{-2}}$ ]")
    # add the gas color bar
    gas_cbar_ax = ax.inset_axes([0.04, 0.1, 0.30, 0.015])
    gas_cbar = fig.colorbar(gas, cax=gas_cbar_ax, pad=0.1, orientation="horizontal")
    gas_cbar_ax.yaxis.set_ticks_position("left")
    gas_cbar_ax.set_title(r"$\Sigma_{\rm gas}$  [$\mathrm{M_\odot \: pc^{-2}}$]")

    # # temp color bar
    # temp_cbar_ax = ax.inset_axes([0.7, 0.84, 0.28, 0.010])
    # temp_cbar = fig.colorbar(temp, cax=temp_cbar_ax, pad=0, orientation="horizontal")
    # temp_cbar_ax.yaxis.set_ticks_position("left")
    # temp_cbar_ax.set_title(r"Temperature \: [K]", fontsize=11)

    ## add time and redshift
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


if __name__ == "__main__":
    mylog.setLevel(40)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    processor_number = 0
    yt.enable_parallelism()
    cell_fields, epf = ram_fields()

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
        print(" rendering gas properties movie ")
        print("********************************************************************")

    datadir = sys.argv[1]
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
        snapshot_type="ramses_snapshot",
    )

    # datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     373,
    #     373,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # render_nickname = "gas_surface_density"

    # =============================================================================
    #                         timelapse paramaters
    # =============================================================================

    pw = 300  # plot width on one side in pc
    r_sf = 500  # radii for sf in pc
    gas_res = 1000  # resolution of the fixed resolution buffer

    # run save
    sim_run = os.path.basename(os.path.normpath(datadir))
    logsfc_path = os.path.join(datadir, "logSFC")
    render_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "renders",
        sim_run,
        render_nickname,
    )
    check_path(render_container)

    ## panning
    zoom_pw = 120
    star_bins = 2000
    pxl_size = (pw / star_bins) ** 2  # pc

    pan_frames = 500  # number of frames to use up for animaton
    num_rots = 4  # number of 360 degreee rotations
    rotation_interval = np.linspace(0, 2 * num_rots, pan_frames, endpoint=False) * np.pi
    zoom_interval = np.concatenate(
        [
            pw * np.ones(int(pan_frames / num_rots)),
            np.linspace(pw, zoom_pw, int(pan_frames / num_rots)),
            zoom_pw * np.ones(int(pan_frames / num_rots)),
            np.linspace(zoom_pw, pw, int(pan_frames / num_rots)),
        ]
    )
    pause_and_rotate = [666]

    for i, fp in enumerate(fpaths):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(fp, f"info_{snums[i]}.txt"))
        print(infofile)
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        t_myr = float(ds.current_time.in_units("Myr"))
        redshift = ds.current_redshift

        if len(np.array(ad["star", "particle_position_x"])) > 0:
            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)
            x_pos = x_pos - x_center
            y_pos = y_pos - y_center
            z_pos = z_pos - z_center

            pop2_xyz = np.array(
                ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")
            ).T

            ctr_at_code = np.array([x_center, y_center, z_center])

            star_mass = np.ones_like(x_pos) * 10
            pop2_xyz = np.array(
                ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")
            ).T
            current_ages = get_star_ages(ram_ds=ds, ram_ad=ad, logsfc=logsfc_path)
            # pop2_lums = lum_look_up_table(
            #     stellar_ages=current_ages * 1e6,
            #     stellar_masses=star_mass,
            #     table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
            #     column_idx=1,
            #     log=False,
            # )
        else:
            print("no stars")
            _, ctr_at_code = ds.find_max(("gas", "density"))
            pop2_lums = np.zeros(100)
            star_mass = np.zeros(100)
            pop2_xyz = np.zeros((100, 3))

        if int(snums[i]) in pause_and_rotate:
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

                stellar_mass_dens, _, _ = np.histogram2d(
                    rotated_star_positions[:, 0],
                    rotated_star_positions[:, 1],
                    bins=star_bins,
                    weights=star_mass,
                    range=[
                        [-plt_wdth / 2, plt_wdth / 2],
                        [-plt_wdth / 2, plt_wdth / 2],
                    ],
                )
                stellar_mass_dens = stellar_mass_dens.T

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
                gas_array = ds.arr(gas_array, "g/cm**2").to("Msun/pc**2")

                temp = yt.OffAxisProjectionPlot(
                    ds,
                    normal=np.dot(np.array([0.0, 0.0, 1.0]), rotation_matrix),
                    fields=("gas", "temperature"),
                    center=ctr_at_code,
                    north_vector=np.array([0.0, 1.0, 0.0]),
                    width=(plt_wdth, "pc"),
                    weight_field=("gas", "density"),
                    # resolution=2000,
                )
                temp_frb = gas.to_fits_data()
                temp_array = np.array(gas_frb[0].data)  # .T

                fig, ax = plt.subplots(
                    figsize=(13, 13),
                    dpi=300,
                    facecolor=cm.Greys_r(0),
                )
                draw_frame(
                    gas_array,
                    temp_array,
                    stellar_mass_dens,
                    ax,
                    fig,
                    plt_wdth,
                    t_myr,
                    redshift,
                    label=render_nickname,
                )
                output_path = os.path.join(
                    render_container,
                    "{}-{}_rotated-{}.png".format(
                        render_nickname, snums[i], str(rot_i).zfill(3)
                    ),
                )
                plt.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
                print(">Saved:", output_path)
                plt.close()
        else:
            plt_wdth = pw
            stellar_mass_dens, _, _ = np.histogram2d(
                pop2_xyz[:, 0],
                pop2_xyz[:, 1],
                bins=star_bins,
                weights=star_mass,
                range=[
                    [-plt_wdth / 2, plt_wdth / 2],
                    [-plt_wdth / 2, plt_wdth / 2],
                ],
            )
            stellar_mass_dens = stellar_mass_dens.T

            gas = yt.ProjectionPlot(
                ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=ctr_at_code
            )
            gas_frb = gas.to_fits_data()
            gas_array = np.array(gas_frb[0].data)  # .T
            gas_array = ds.arr(gas_array, "g/cm**2").to("Msun/pc**2")

            temp = yt.ProjectionPlot(
                ds,
                "z",
                ("gas", "temperature"),
                width=(plt_wdth, "pc"),
                center=ctr_at_code,
                weight_field=("gas", "density"),
            )
            temp_frb = temp.data_source.to_frb((plt_wdth, "pc"), star_bins)
            temp_array = np.array(temp_frb["gas", "temperature"])

            fig, ax = plt.subplots(figsize=(13, 13), dpi=400, facecolor=cm.Greys_r(0))
            draw_frame(
                gas_array,
                temp_array,
                stellar_mass_dens,
                ax,
                fig,
                plt_wdth,
                t_myr,
                redshift,
                label=render_nickname,
            )
            output_path = os.path.join(
                render_container, "{}-{}.png".format(render_nickname, snums[i])
            )
            # plt.show()
            plt.savefig(
                os.path.expanduser(output_path),
                dpi=250,
                bbox_inches="tight",
                pad_inches=0.05,
            )

            print(">>> saved:", output_path)
            plt.show()
