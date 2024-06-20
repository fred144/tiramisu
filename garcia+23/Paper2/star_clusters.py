import sys

sys.path.append("../../")
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
import matplotlib as mpl
from tools.fscanner import filter_snapshots
from tools.ram_fields import ram_fields
import warnings
import cmasher as cmr
from src.lum.lum_lookup import lum_look_up_table
from src.lum.pop2 import get_star_ages
from matplotlib.colors import LinearSegmentedColormap
from astropy import units as u

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.size": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    }
)


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
    luminosity,
    ax,
    fig,
    wdth,
    t_myr,
    redshift,
    label,
    star_bins=2000,
    gauss_sig=9,
):
    lum_range = (6e33, 3e36)
    gas_range = (0.005, 0.32)
    temp_range = (2e3, 2e5)

    pxl_size = (wdth / star_bins) ** 2
    lum_alpha = 1

    surface_brightness = luminosity / pxl_size
    # clean up edges

    ax.axis("off")

    # dictate alphas manually
    # decide which part of the cmap is increasing in alpha
    # the final transparencey is dictatted by

    gascmap = linear_cmap(0.1, 0.6, "mpl", "cubehelix")
    tempcmap = linear_cmap(0.05, 0.3, "cmr", "cmr.gothic")
    lumcmap = cmr.amethyst

    lum = ax.imshow(
        surface_brightness,
        cmap=lumcmap,
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
        alpha=lum_alpha,
    )
    temp = ax.imshow(
        gaussian_filter(temp_array, gauss_sig),
        cmap=tempcmap,
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(temp_range[0], temp_range[1]),
    )
    gas = ax.imshow(
        gaussian_filter(
            (gas_array * (u.g / u.cm**2)).to(u.Msun / u.pc**2).value, gauss_sig
        ),
        cmap=gascmap,
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(
            (gas_range[0] * (u.g / u.cm**2)).to(u.Msun / u.pc**2).value,
            (gas_range[1] * (u.g / u.cm**2)).to(u.Msun / u.pc**2).value,
        ),
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

    ##add the luminosity color bar
    lum_cbar_ax = ax.inset_axes([0.19, 0.03, 0.010, 0.18])
    lum_cbar = fig.colorbar(lum, cax=lum_cbar_ax, pad=0)
    # lum_cbar_ax.yaxis.set_ticks_position("left")
    lum_cbar_ax.set_ylabel(
        r" $I_\lambda \: $ [${\rm erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}}$]",
        fontsize=11,
    )
    # add the gas color bar
    gas_cbar_ax = ax.inset_axes([0.05, 0.03, 0.010, 0.18])
    gas_cbar = fig.colorbar(gas, cax=gas_cbar_ax, pad=0)
    # gas_cbar_ax.yaxis.set_ticks_position("left")
    gas_cbar_ax.set_ylabel(
        r"Surface Density [$\mathrm{ M_\odot \: pc^{-2}}$]", fontsize=11
    )
    # temp
    temp_cbar_ax = ax.inset_axes([0.12, 0.03, 0.010, 0.18])
    temp_cbar = fig.colorbar(temp, cax=temp_cbar_ax, pad=0)
    # gas_cbar_ax.yaxis.set_ticks_position("left")
    temp_cbar_ax.set_ylabel(
        r"WeightedTemperature $\mathrm{ \left[K\right]}$", fontsize=11
    )

    ##add time and redshift
    ax.text(
        0.05,
        0.96,
        (
            "\n" r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$"
        ).format(
            # label,
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
    plt.style.use("dark_background")

    # if len(sys.argv) != 6:
    #     print(sys.argv[0], "usage:")
    #     print(
    #         "{} snapshot_dir start_snap end_snap step render_nickname".format(
    #             sys.argv[0]
    #         )
    #     )
    #     exit()
    # else:
    #     print("********************************************************************")
    #     print(" rendering gas properties movie ")
    #     print("********************************************************************")

    # datadir = sys.argv[1]
    # start_snapshot = int(sys.argv[2])
    # end_snapshot = int(sys.argv[3])
    # step = int(sys.argv[4])
    # render_nickname = sys.argv[5]

    # sim_run = os.path.basename(os.path.normpath(datadir))
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     start_snapshot,
    #     end_snapshot,
    #     sampling=step,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    ## local test

    datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    fpaths, snums = filter_snapshots(
        datadir,
        303,
        303,
        sampling=1,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )
    render_nickname = "test"

    # =============================================================================
    #                         timelapse paramaters
    # =============================================================================

    pw = 640  # plot width on one side in pc
    r_sf = 500  # radii for sf in pc
    gas_res = 1000  # resolution of the fixed resolution buffer

    # run save
    sim_run = os.path.basename(os.path.normpath(datadir))
    logsfc_path = os.path.join(datadir, "logSFC")
    render_container = os.path.join(
        "..",
        "..",
        "..",
        "container_tiramisu",
        "renders",
        sim_run,
        render_nickname,
    )
    check_path(render_container)

    ## panning
    zoom_pw = 150
    star_bins = 2000
    pxl_size = (pw / star_bins) ** 2  # pc

    pan_frames = 400  # number of frames to use up for animaton
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
    pause_and_rotate = [301]

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
            pop2_lums = lum_look_up_table(
                stellar_ages=current_ages * 1e6,
                stellar_masses=star_mass,
                table_link=os.path.join("..", "..", "starburst", "l1500_inst_e.txt"),
                column_idx=1,
                log=False,
            )
        else:
            _, ctr_at_code = ds.find_max(("gas", "density"))

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

                lums, _, _ = np.histogram2d(
                    rotated_star_positions[:, 0],
                    rotated_star_positions[:, 1],
                    bins=star_bins,
                    weights=pop2_lums,
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
                temp_frb = temp.to_fits_data()
                temp_array = np.array(temp_frb[0].data)  # .T

                fig, ax = plt.subplots(
                    figsize=(13, 13),
                    dpi=300,
                    facecolor=cm.Greys_r(0),
                )
                draw_frame(
                    gas_array,
                    temp_array,
                    lums,
                    ax,
                    fig,
                    plt_wdth,
                    t_myr,
                    redshift,
                    label=sim_run,
                    gauss_sig=5,
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
            lums, _, _ = np.histogram2d(
                pop2_xyz[:, 0],
                pop2_xyz[:, 1],
                bins=star_bins,
                weights=pop2_lums,
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
                lums,
                ax,
                fig,
                plt_wdth,
                t_myr,
                redshift,
                label=sim_run,
                gauss_sig=9,
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
            plt.close()
