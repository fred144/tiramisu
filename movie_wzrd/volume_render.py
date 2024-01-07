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

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0


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
    gas_cbar_ax.set_ylabel(r"$\mathrm{ g \: cm^{-2}}$", fontsize=10, labelpad=-22)

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
    yt.enable_parallelism()
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
    #     print(" rendering movie ")
    #     print("********************************************************************")

    # datadir = sys.argv[1]
    # start_snapshot = int(sys.argv[2])
    # end_snapshot = int(sys.argv[3])
    # step = int(sys.argv[4])
    # render_nickname = sys.argv[5]

    ## local path for test
    # datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")
    # datadir = os.path.relpath("../../garcia23_testdata/fs07_refine")
    datadir = os.path.expanduser("~/test_data/fs035_ms10/")
    start_snapshot = 567
    end_snapshot = 567
    step = 1
    render_nickname = "test"

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
    pause_and_rotate = [304, 405]

    for i, sn in enumerate(snaps):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        t_myr = float(ds.current_time.in_units("Myr"))
        redshift = ds.current_redshift

        if len(np.array(ad["star", "particle_position_x"])) > 0:
            x_pos = ad["star", "particle_position_x"]
            y_pos = ad["star", "particle_position_y"]
            z_pos = ad["star", "particle_position_z"]
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)
            x_pos = x_pos - x_center
            y_pos = y_pos - y_center
            z_pos = z_pos - z_center

            ctr_at_code = ds.arr([x_center, y_center, z_center])
        else:
            _, ctr_at_code = ds.find_max(("gas", "density"))

        bounds = (1e-23, 2.366763e-20)
        width = (1000, "pc")
        sp = ds.sphere(ctr_at_code, width)
        data_source = ds.sphere(sp.center, (width[0] * 4, "pc"))

        sc = yt.create_scene(
            data_source, lens_type="perspective", field=("gas", "density")
        )

        sc.camera.resolution = (1024, 1024)
        sc.camera.focus = data_source.argmax("density")
        sc.camera.position = sc.camera.focus + ds.quan(width[0] * 4, "pc")

        source = sc[0]

        rho_min, rho_max = ds.arr([1e-3, 1e2], "mp/cm**3").to("g/cm**3")
        source.tfh.set_bounds((rho_min, rho_max))
        source.tfh.grey_opacity = True
        source.tfh.plot("transfer_function.png", profile_field="density")

        source.tfh.plot()
        sc.render()
        sc.show(sigma_clip=8.0)

        # cam_position = [
        #     float(ctr_at_code[0]),
        #     float(ctr_at_code[1]),
        #     float(ctr_at_code[2]) - float(ds.arr(*width).to("code_length")),
        # ]

        # sc = yt.create_scene(ds)
        # cam = sc.add_camera()
        # cam.focus = np.array(ctr_at_code)
        # # cam.position = np.array(cam_position)
        # cam.switch_orientation()

        # source = sc[0]
        # source.set_field(("gas", "density"))
        # source.set_log(True)

        # # Since this rendering is done in log space, the transfer function needs
        # # to be specified in log space.
        # tf = yt.ColorTransferFunction(np.log10(bounds))

        # def linramp(vals, minval, maxval):
        #     return (vals - vals.min()) / (vals.max() - vals.min())

        # tf.map_to_colormap(
        #     np.log10(bounds[0]),
        #     np.log10(bounds[1]),
        #     colormap="cmyt.arbre",
        #     scale_func=linramp,
        # )

        # source.tfh.tf = tf
        # source.tfh.bounds = bounds

        # source.tfh.plot("transfer_function.png", profile_field=("gas", "density"))
        # im = sc.render()
        # sc.save("rendering.png")
