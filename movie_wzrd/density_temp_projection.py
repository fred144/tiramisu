import sys


sys.path.append("..")  # makes sure that importing the modules work
import numpy as np
import os
import glob
from src.lum.lum_lookup import lum_look_up_table
from scipy.ndimage import gaussian_filter

from tools.cosmo import t_myr_from_z, code_age_to_myr
from tools.ram_fields import ram_fields
from tools.fscanner import filter_snapshots

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import yt

#%%
yt.enable_parallelism()
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "font.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 1,
    }
)
plt.style.use("dark_background")

#%%
cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
plt_wdth = 400

star_bins = 2000
pxl_size = (plt_wdth / star_bins) ** 2  # pc

star_t_range = (355, 662)
gas_alpha = 0.5
lum_alpha = 1

cell_fields, epf = ram_fields()
datadir = os.path.relpath("../../cosm_test_data/refine")
# datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")


snaps, snap_strings = filter_snapshots(datadir, 500, 500, sampling=2, str_snaps=True)


movie_name = "ProjLum"
sim_run = datadir.replace("\\", "/").split("/")[-1]

container = os.path.join(
    "..", "..", "container_tiramisu", "renders", movie_name, sim_run
)

if not os.path.exists(container):
    print("Creating container", container)
    os.makedirs(container)
else:
    pass
sim_logfile = os.path.join(
    "..",
    "..",
    "container_tiramisu",
    "sim_log_files",
    "fs07_refine" if sim_run == "refine" else sim_run,
    "logSFC",
)

for i, sn in enumerate(snaps):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
    print("# reading in", infofile)

    ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()
    hubble = ds.hubble_constant
    tmyr = float(ds.current_time.in_units("Myr"))
    zred = float(ds.current_redshift)

    if len(ad["star", "particle_position_x"]) == 0:
        print("no popii yet")
        # continue
        # stars havent formed yet
        # find CoM of the system, starting from the most dense gas coord
        gal = ds.sphere("max", (plt_wdth / 2, "pc"))
        # return CoM in code units
        com = gal.quantities.center_of_mass(
            use_gas=True, use_particles=True, particle_type="star"
        )

        # recenter the stars based on the CoM
        x = np.array((ad["star", "particle_position_x"] - com[0]).to("pc"))
        y = np.array((ad["star", "particle_position_y"] - com[1]).to("pc"))
        z = np.array((ad["star", "particle_position_z"] - com[2]).to("pc"))

        lums = np.zeros((star_bins, star_bins))
    else:
        com_x = np.mean(ad["star", "particle_position_x"])
        com_y = np.mean(ad["star", "particle_position_y"])
        com_z = np.mean(ad["star", "particle_position_z"])

        com = np.array([com_x, com_y, com_z])

        x = np.array((ad["star", "particle_position_x"] - com_x).to("pc"))
        y = np.array((ad["star", "particle_position_y"] - com_y).to("pc"))
        z = np.array((ad["star", "particle_position_z"] - com_z).to("pc"))

        unique_birth_epochs = code_age_to_myr(
            ad["star", "particle_birth_epoch"], hubble, unique_age=True
        )
        # calculate the age of the universe when the first star was born
        # using the logSFC as a reference point for redshift when the first star
        # was born. Every age is relative to this. Due to our mods of ramses.
        first_form = np.loadtxt(sim_logfile, usecols=2).max()
        birth_start = np.round_(
            float(ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
        )
        # all the birth epochs of the stars
        converted_unfiltered = code_age_to_myr(
            ad["star", "particle_birth_epoch"], hubble, unique_age=False
        )
        abs_birth_epochs = np.round(converted_unfiltered + birth_start, 3)  #!
        current_ages = np.round(tmyr, 3) - np.round(abs_birth_epochs, 3)

        #%%
        print("Looking Up Lums")
        star_lums = (
            lum_look_up_table(
                stellar_ages=current_ages,
                table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
                column_idx=1,
            )
            * 1e-5
        )

        print("Integrating Luminosity")
        # get the projected luminosity
        lums, _, _ = np.histogram2d(
            x,
            y,
            bins=star_bins,
            weights=np.log10(star_lums),
            normed=False,
            range=[
                [-plt_wdth / 2, plt_wdth / 2],
                [-plt_wdth / 2, plt_wdth / 2],
            ],
        )
        lums = lums.T
    #%%
    # f3_unique_birth_times = np.unique(f3_rounded_times)
    # f3_unique_birth_times = np.unique(f3_rounded_times)
    # get the projected densities
    #!!!
    print("Integrating Gas")
    gas = yt.ProjectionPlot(
        ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=com
    )
    gas_frb = gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
    gas_array = np.array(gas_frb["gas", "density"])

    print("Getting Temp")
    t = yt.ProjectionPlot(
        ds,
        "z",
        ("gas", "temperature"),
        width=(plt_wdth, "pc"),
        center=com,
        weight_field=("gas", "density"),
    )
    temp_frb = t.data_source.to_frb((plt_wdth, "pc"), star_bins)
    temp_array = np.array(temp_frb["gas", "temperature"])

    lum_range = (33.3, 36.477)  # (2e32, 5e35)
    gas_range = (0.008, 0.30)
    temp_range = (6e3, 3e5)

    ncolors = 256
    gas_cmap_arr = plt.get_cmap("cubehelix")(range(ncolors))
    # dictate alphas manually
    # decide which part of the cmap is increasing in alpha
    # the final transparencey is dictatted by
    final_trans_gas = 0.65
    gas_cmap_arr[:150, -1] = np.linspace(
        0.0, final_trans_gas, gas_cmap_arr[:150, -1].size
    )
    gas_cmap_arr[150:, -1] = np.ones(gas_cmap_arr[150:, -1].size) * final_trans_gas
    gascmap = LinearSegmentedColormap.from_list(
        name="cubehelix_alpha", colors=gas_cmap_arr
    )
    final_trans_tem = 0.65
    temp_cmap_arr = plt.get_cmap("gnuplot2")(range(ncolors))
    temp_cmap_arr[:200, -1] = np.linspace(
        0.0, final_trans_tem, temp_cmap_arr[:200, -1].size
    )
    temp_cmap_arr[200:, -1] = np.ones(temp_cmap_arr[200:, -1].size) * final_trans_tem
    tempcmap = LinearSegmentedColormap.from_list(
        name="dusk_alpha", colors=temp_cmap_arr
    )
    # # register this new colormap with matplotlib
    # plt.colormaps.register(cmap=map_object)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(6, 6),
        dpi=400,
        facecolor=cm.Greys_r(0),
    )

    lum_image = ax.imshow(
        lums,
        cmap="inferno",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=1,
    )
    # gas_image = ax.imshow(
    #     gaussian_filter(
    #         np.log10(
    #             gas_array,
    #             where=(gas_array != 0),
    #             out=np.full_like(lums, np.log10(gas_range[0])),
    #         ),
    #         8,
    #     ),
    #     origin="lower",
    #     extent=[
    #         -plt_wdth / 2,
    #         plt_wdth / 2,
    #         -plt_wdth / 2,
    #         plt_wdth / 2,
    #     ],
    #     vmin=np.log10(gas_range[0]),
    #     vmax=np.log10(gas_range[1]),
    #     cmap=gascmap,
    # )
    # temp_image = ax.imshow(
    #     gaussian_filter(
    #         np.log10(
    #             temp_array,
    #             where=(temp_array != 0),
    #             out=np.full_like(temp_array, 0),
    #         ),
    #         8,
    #     ),
    #     origin="lower",
    #     extent=[
    #         -plt_wdth / 2,
    #         plt_wdth / 2,
    #         -plt_wdth / 2,
    #         plt_wdth / 2,
    #     ],
    #     vmin=np.log10(temp_range[0]),
    #     vmax=np.log10(temp_range[1]),
    #     cmap=tempcmap,
    # )

    lum_cbar_ax = ax.inset_axes([0.05, 0.05, 0.30, 0.028], alpha=0.8)
    lum_cbar = fig.colorbar(lum_image, cax=lum_cbar_ax, pad=0, orientation="horizontal")
    lum_cbar.ax.xaxis.set_tick_params(pad=2)
    lum_cbar.set_label(
        r"$\mathrm{\log\:UV\:SB}$" r"$\:\mathrm{\left(erg\:\:s^{-1}\:pc^{-2}\right)}$",
        labelpad=-15,
    )

    # gas_cbar_ax = ax.inset_axes([0.35, 0.05, 0.30, 0.028], alpha=0.8)
    # gas_cbar = fig.colorbar(
    #     cm.ScalarMappable(
    #         norm=mpl.colors.Normalize(np.log10(gas_range[0]), np.log10(gas_range[1])),
    #         cmap=gascmap,
    #     ),
    #     cax=gas_cbar_ax,
    #     orientation="horizontal",
    #     pad=0,
    # )
    # gas_cbar.ax.xaxis.set_tick_params(pad=2)
    # gas_cbar.set_label(
    #     label=r"$\mathrm{\log\:Gas\:Density\:\left(g \: cm^{-2}\right)}$", labelpad=-15
    # )

    # temp_cbar_ax = ax.inset_axes([0.65, 0.05, 0.30, 0.028], alpha=0.8)
    # temp_cbar = fig.colorbar(
    #     cm.ScalarMappable(
    #         norm=mpl.colors.Normalize(np.log10(temp_range[0]), np.log10(temp_range[1])),
    #         cmap=tempcmap,
    #     ),
    #     cax=temp_cbar_ax,
    #     orientation="horizontal",
    #     pad=0,
    # )
    # temp_cbar.ax.xaxis.set_tick_params(pad=2)
    # temp_cbar.set_label(label=r"$\mathrm{\log\:Gas\:Temperature\:(K)}$", labelpad=-15)

    # clean up edges
    ax.set(
        xlim=(-plt_wdth / 2, plt_wdth / 2),
        ylim=(-plt_wdth / 2, plt_wdth / 2),
        xticklabels=[],
        yticklabels=[],
    )
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # add some scale bar
    scale = patches.Rectangle(
        xy=(-plt_wdth / 2 * 0.25, plt_wdth / 2 * 0.89),
        width=plt_wdth / 2 * 0.5,
        height=0.010 * plt_wdth / 2,
        linewidth=0,
        edgecolor="white",
        facecolor="white",
    )
    ax.text(
        0,
        plt_wdth / 2 * 0.93,
        r"$\mathrm{{{:.0f} \: pc}}$".format(plt_wdth / 2 * 0.5),
        ha="center",
        va="center",
        color="white",
        # fontproperties=leg_font,
    )
    ax.add_patch(scale)

    # time  and redshift
    ax.text(
        0.05,
        0.95,
        (r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$").format(
            tmyr, zred
        ),
        ha="left",
        va="center",
        color="white",
        transform=ax.transAxes,
    )
    # efficiency labels
    if sim_run == "fs035_ms10":
        plt_run_label = r"$\mathrm{low-SFE\:(35\%)}$"
    elif sim_run == "fs07_refine":
        plt_run_label = r"$\mathrm{low-SFE\:(70\%)}$"
    else:
        plt_run_label = sim_run
    ax.text(
        s=plt_run_label,
        x=0.95,
        y=0.95,
        ha="right",
        va="center",
        color="white",
        transform=ax.transAxes,
    )

    # save frame
    ax.axis("off")
    output_path = os.path.join(
        container,
        r"render-{}-{}.png".format(
            snap_strings[i], "{:.3f}".format(zred).replace(".", "_")
        ),
    )
    plt.savefig(
        os.path.expanduser(output_path), dpi=400, bbox_inches="tight", pad_inches=0.00
    )
    # plt.close("all")
    print(">Saved:", output_path)
