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
from src.lum.lum_lookup import lum_look_up_table
from tools.cosmo import code_age_to_myr
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage import gaussian_filter
from tools.cosmo import t_myr_from_z
from scipy import interpolate

# %%
mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": False,
        "xtick.top": False,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.size": 14,
    }
)


def get_star_ages(ram_ds, ram_ad, logsfc):
    """
    star's ages in [Myr]
    """
    first_form = np.loadtxt(logsfc, usecols=2).max()
    current_hubble = ram_ds.hubble_constant
    current_time = float(ram_ds.current_time.in_units("Myr"))

    birth_start = np.round_(
        float(ram_ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
    )
    converted_unfiltered = code_age_to_myr(
        ram_ad["star", "particle_birth_epoch"],
        current_hubble,
        unique_age=False,
    )
    birthtime = np.round(converted_unfiltered + birth_start, 3)  #!
    current_ages = np.array(np.round(current_time, 3) - np.round(birthtime, 3))
    return current_ages


def draw_frame(
    gas_array,
    luminosity,
    ax,
    fig,
    wdth,
    t_myr,
    redshift,
    sf_data,
    label,
    star_bins=2000,
):
    msun_pc2 = 4786.79
    gas_array = gaussian_filter(gas_array * msun_pc2, 9)

    lum_range = (3e33, 3e36)
    sigma_gas_range = (0.008 * msun_pc2, 0.32 * msun_pc2)
    pxl_size = (wdth / star_bins) ** 2
    lum_alpha = 1
    gas_alpha = 0.55
    surface_brightness = luminosity / pxl_size
    # clean up edges
    ax.axis("off")

    # luminosity and gas density
    lum = ax.imshow(
        surface_brightness,
        cmap="inferno",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
        alpha=lum_alpha,
    )
    gas = ax.imshow(
        gas_array,
        cmap="cubehelix",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(sigma_gas_range[0], sigma_gas_range[1]),
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
    lum_cbar_ax = ax.inset_axes([1.06, 0.05, 0.010, 0.40])
    lum_cbar = fig.colorbar(lum, cax=lum_cbar_ax, pad=0)
    lum_cbar_ax.set_ylabel(r"${\rm erg\:\:s^{-1}\:\AA^{-1}\:pc^{-2}}$", fontsize=14)
    # add the gas color bar
    gas_cbar_ax = ax.inset_axes([1.045, 0.05, 0.010, 0.40])
    gas_cbar = fig.colorbar(gas, cax=gas_cbar_ax, pad=0)
    gas_cbar_ax.yaxis.set_ticks_position("left")
    gas_cbar_ax.set_ylabel(
        r"$\Sigma_\mathrm{gas}\:\mathrm{(M_\odot\:pc^{-2})}$", fontsize=14, labelpad=-45
    )

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

    # add zoom axis
    xw = 0.7
    yh = 0.45
    z_mult = 0.05
    r = yh / xw
    zoom = ax.inset_axes([1.02, 0.50, xw, yh])
    zoom.xaxis.tick_top()
    zoom.xaxis.set_label_position("top")
    zoom.imshow(
        surface_brightness,
        cmap="inferno",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(vmin=lum_range[0], vmax=lum_range[1]),
        alpha=lum_alpha,
    )
    zoom.imshow(
        gas_array,
        cmap="cubehelix",
        interpolation="gaussian",
        origin="lower",
        extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
        norm=LogNorm(sigma_gas_range[0], sigma_gas_range[1]),
        alpha=gas_alpha,
    )

    zoom.set(
        xlim=(-wdth / 2 * z_mult, wdth / 2 * z_mult),
        ylim=(-wdth / 2 * z_mult * r, wdth / 2 * z_mult * r),
        # xticklabels=[],
        yticklabels=[],
        xlabel="parsec",
    )
    rect, lines = ax.indicate_inset_zoom(zoom)
    lines[1].set_alpha(0)
    lines[2].set_alpha(0)
    rect.set_linewidth(1.0)
    # mark_inset(ax, zoom, loc1=2, loc2=2, edgecolor="white", alpha=0.2, lw=1)

    sfr_ax = ax.inset_axes([1.22, 0.25, 0.5, 0.2])
    sfh_ax = ax.inset_axes([1.22, 0.05, 0.5, 0.2])

    sfr_ax.spines["top"].set_visible(False)
    sfr_ax.spines["right"].set_visible(False)
    sfh_ax.spines["right"].set_visible(False)

    t, sfh, sfr = sf_data
    tmask = t <= t_myr

    cmap = matplotlib.colormaps["magma"]
    cmap = cmap(np.linspace(0, 1, 100))

    sfr_ax.plot(t[tmask], sfr[tmask], lw=3, alpha=0.8, color="orchid")
    sfr_ax.plot(t[~tmask], sfr[~tmask], lw=3, alpha=0.5, color="grey")

    sfh_ax.plot(t[tmask], sfh[tmask], lw=3, alpha=0.8, color="crimson")
    sfh_ax.plot(t[~tmask], sfh[~tmask], lw=3, alpha=0.5, color="grey")

    sfr_ax.set(
        ylabel=r"$\mathrm{SFR\:\left(M_{\odot}\:yr^{-1}\right)}$",
        xlim=(t.min(), t.max()),
    )
    sfh_ax.set(
        ylabel=r"Total PopII Mass $(\rm{M_{\odot}})$",
        xlabel="$\mathrm{time\:(Myr)}$",
        yscale="log",
        xlim=(t.min(), t.max()),
        ylim=(1e3, 2 * np.max(sfh)),
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
    # render_nickname = "sfr_test"

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
    logsfc_container = os.path.join(
        "..", "..", "container_tiramisu", "sim_log_files", sim_run
    )
    cell_fields, epf = ram_fields()

    # sfr and starformation history sfh
    sim_logsfc = np.loadtxt(os.path.join(logsfc_container, "logSFC"))
    z = sim_logsfc[:, 2]
    t = t_myr_from_z(z)
    mass_in_star = sim_logsfc[:, 7]
    running_total_mass = np.cumsum(mass_in_star)
    sfr_binwidth_myr = 1
    # interpolate points so that SFR is not infinity, since SF is a step function
    t_interp_points = np.arange(t.min(), t.max(), sfr_binwidth_myr)
    total_mass_interpolator = interpolate.interp1d(
        x=t, y=running_total_mass, kind="previous"
    )
    total_mass = total_mass_interpolator(t_interp_points)
    # calculate the sfr in msun / yr
    sfr = np.gradient(total_mass) / (sfr_binwidth_myr * 1e6)
    sf_data = (t_interp_points, total_mass, sfr)

    # =============================================================================
    #                         timelapse parameters
    # =============================================================================
    static_plt_wdth = 400
    zoom_plt_wdth = 120
    star_bins = 2000
    pxl_size = (static_plt_wdth / star_bins) ** 2  # pc

    pause_and_rotate = [100]
    pan_frames = 200  # number of frames to use up for each rotation,
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

    for i, sn in enumerate(snaps):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)

        # check if there is post processed data
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
            pop2_lums = 10 ** pop2_data[:, 3]
            pop2_xyz = pop2_data[:, 4:7]  # pc
            # pop2_y = pop2_data[:, 5]
            # pop2_z = pop2_data[:, 6]
        else:  # calculate on the fly
            print("*** no post-processed pop2")
            ad = ds.all_data()
            # get time-dependent params.
            redshft = ds.current_redshift
            current_hubble = ds.hubble_constant
            current_time = float(ds.current_time.in_units("Myr"))
            # read POPII star info
            star_id = np.array(ad["star", "particle_identity"])
            star_mass = np.ones_like(star_id) * 10  # uniform 10 solar mass
            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            if len(star_id) == 0:
                print("- no particle data to extract")
                continue
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)

            ctr_at_code = np.array([x_center, y_center, z_center])
            # translate points to stellar CoM
            x_pos = x_pos - x_center
            y_pos = y_pos - y_center
            z_pos = z_pos - z_center
            pop2_xyz = np.array(
                ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")
            ).T

            current_ages = get_star_ages(ram_ds=ds, ram_ad=ad, logsfc=sim_logsfc)
            pop2_lums = lum_look_up_table(
                stellar_ages=current_ages * 1e6,  # in myr
                stellar_masses=star_mass,
                table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
                column_idx=1,
                log=False,
            )

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
                    weights=pop2_lums,
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

            # zoom_gas = yt.ProjectionPlot(
            #     ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=ctr_at_code
            # )
            # zoom_gas_frb = zoom_gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
            # zoom_gas_array = np.array(zoom_gas_frb["gas", "density"])

            fig, ax = plt.subplots(figsize=(13, 13), dpi=400, facecolor=cm.Greys_r(0))
            draw_frame(
                gas_array,
                lums,
                ax,
                fig,
                plt_wdth,
                t_myr,
                redshift,
                sf_data,
                label=sim_run,
            )
            output_path = os.path.join(
                render_container, "{}-{}.png".format(render_nickname, snap_strings[i])
            )
            # plt.show()
            plt.savefig(
                os.path.expanduser(output_path),
                dpi=300,
                bbox_inches="tight",
                # pad_inches=0.05,
            )

            print(">>> saved:", output_path)
            # plt.close()
