import sys

sys.path.insert(1, "/homes/fgarcia4/py-virtual-envs/master/lib/python3.7/site-packages")
sys.path.append("..")  # makes sure that importing the modules work
import numpy as np
import os
import glob
from src.lum import lum_lookup
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
# datadir = os.path.relpath("../../sim_data/cluster_evolution/fs035_ms10")


snaps, snap_strings = filter_snapshots(datadir, 500, 500, sampling=1, str_snaps=True)


movie_name = "ProjDensTemp"
sim_run = datadir.replace("\\", "/").split("/")[-1]

container = os.path.join("..", "..", "container_ram-py", "renders", movie_name, sim_run)
if not os.path.exists(container):
    print("Creating ram-py container", container)
    os.makedirs(container)
for i, sn in enumerate(snaps):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
    print("# reading in", infofile)

    ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()
    hubble = ds.hubble_constant
    tmyr = float(ds.current_time.in_units("Myr"))
    zred = float(ds.current_redshift)
#%% can the code above is set up to work with time matching
# but not used since single frame

for m_i, (f7_gas, f3_gas) in enumerate(zip(f7_snap_f, f3_snap_f)):
    outnum_f7 = f7_gas.split("/")[-1].split("_")[-1]
    outnum_f3 = f3_gas.split("/")[-1].split("_")[-1]

    #!!!
    if int(outnum_f3) == 1102:
        continue
    # read ramses data
    print("Reading", f3_gas)
    f3_info_file = os.path.join(f3_gas, "info_{}.txt".format(outnum_f3))
    f3_ram_ds = yt.load(f3_info_file, fields=cell_fields, extra_particle_fields=epf)
    # post processed star data
    f3_code_ctr = np.loadtxt(f3_pop2_f[m_i], max_rows=5)[2:5, 6]
    f3_t_myr = np.loadtxt(f3_pop2_f[m_i], max_rows=2)[0, 6]
    f3_redshift = np.loadtxt(f3_pop2_f[m_i], max_rows=2)[1, 6]
    try:
        f3_stars = np.vstack(
            (
                np.loadtxt(os.path.join(f3_halo_f[m_i], "field_stars.txt")),
                np.loadtxt(os.path.join(f3_halo_f[m_i], "bound_stars.txt")),
            )
        )
        f3_star_ids = f3_stars[:, 0]
        f3_star_lums = f3_stars[:, 2]
        f3_x = f3_stars[:, 3]
        f3_y = f3_stars[:, 4]
        f3_z = f3_stars[:, 5]
        f3_current_ages = f3_stars[:, 1]
        f3_star_bes = f3_t_myr - f3_current_ages
        f3_rounded_times = np.round_(f3_star_bes, 1)
        f3_unique_birth_times = np.unique(f3_rounded_times)
    except:
        print("does not exist, creating luminosity tables")
        f3_current_hubble = f3_ram_ds.hubble_constant
        f3_ad = f3_ram_ds.all_data()
        f3_t_myr = float(f3_ram_ds.current_time.in_units("Myr"))
        f3_redshift = float(f3_ram_ds.current_redshift)

        # find CoM of the system, starting from the most dense gas coord
        f3_sphere = f3_ram_ds.sphere("max", (plt_wdth / 2, "pc"))
        # return CoM in code units
        f3_com = f3_sphere.quantities.center_of_mass(
            use_gas=True, use_particles=True, particle_type="star"
        )

        # recenter the stars based on the CoM
        f3_x = np.array((f3_ad["star", "particle_position_x"] - f3_com[0]).to("pc"))
        f3_y = np.array((f3_ad["star", "particle_position_y"] - f3_com[1]).to("pc"))
        f3_z = np.array((f3_ad["star", "particle_position_z"] - f3_com[2]).to("pc"))
        f3_be_star = f3_ad["star", "particle_birth_epoch"]
        f3_unique_birth_epochs = code_age_to_myr(
            f3_ad["star", "particle_birth_epoch"], f3_current_hubble, unique_age=True
        )
        # calculate the age of the universe when the first star was born
        # using the logSFC as a reference point for redshift when the first star
        # was born. Every age is relative to this. Due to our mods of ramses.
        f3_birth_start = np.round_(
            float(f3_ram_ds.cosmology.t_from_z(f3_series[0, 2]).in_units("Myr")), 0
        )
        # all the birth epochs of the stars
        f3_converted_unfiltered = code_age_to_myr(
            f3_ad["star", "particle_birth_epoch"], f3_current_hubble, unique_age=False
        )
        f3_abs_birth_epochs = np.round(f3_converted_unfiltered + f3_birth_start, 3)  #!
        f3_current_ages = np.round(f3_t_myr, 3) - np.round(f3_abs_birth_epochs, 3)

        f3_star_lums = (
            lum_lookup(
                stellar_ages=f3_current_ages,
                table_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
                column_idx=1,
                log=True,
            )
            * 1e-5
        )

        # f3_unique_birth_times = np.unique(f3_rounded_times)
        # f3_unique_birth_times = np.unique(f3_rounded_times)
    # get the projected densities
    #!!!
    print("Integrating Gas")
    f3_gas = yt.ProjectionPlot(
        f3_ram_ds, "z", ("gas", "density"), width=(plt_wdth, "pc"), center=f3_code_ctr
    )
    f3_gas_frb = f3_gas.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f3_gas_array = np.array(f3_gas_frb["gas", "density"])

    print("Getting Temp")
    f3_t = yt.ProjectionPlot(
        f3_ram_ds,
        "z",
        ("gas", "temperature"),
        width=(plt_wdth, "pc"),
        center=f3_code_ctr,
        weight_field=("gas", "density"),
    )
    f3_temp_frb = f3_t.data_source.to_frb((plt_wdth, "pc"), star_bins)
    f3_temp_array = np.array(f3_temp_frb["gas", "temperature"])

    print("Integrating Luminosity")
    # get the projected luminosity
    f3_lums, _, _ = np.histogram2d(
        f3_x,
        f3_y,
        bins=star_bins,
        weights=f3_star_lums,
        normed=False,
        range=[
            [-plt_wdth / 2, plt_wdth / 2],
            [-plt_wdth / 2, plt_wdth / 2],
        ],
    )
    f3_lums = f3_lums.T

    #%%
    lum_range = (2e33, 3e36)  # (2e32, 5e35)
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

    f3_lum_image = ax.imshow(
        np.log10(
            f3_lums / pxl_size,
            where=(f3_lums != 0),
            out=np.full_like(f3_lums, np.log10(lum_range[0])),
        ),
        cmap="inferno",
        origin="lower",
        extent=[-plt_wdth / 2, plt_wdth / 2, -plt_wdth / 2, plt_wdth / 2],
        vmin=np.log10(lum_range[0]),
        vmax=np.log10(lum_range[1]),
        alpha=1,
    )
    f3_gas_image = ax.imshow(
        gaussian_filter(
            np.log10(
                f3_gas_array,
                where=(f3_gas_array != 0),
                out=np.full_like(f3_lums, np.log10(gas_range[0])),
            ),
            8,
        ),
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(gas_range[0]),
        vmax=np.log10(gas_range[1]),
        cmap=gascmap,
    )
    f3_temp_image = ax.imshow(
        gaussian_filter(
            np.log10(
                f3_temp_array,
                where=(f3_temp_array != 0),
                out=np.full_like(f3_temp_array, 0),
            ),
            8,
        ),
        origin="lower",
        extent=[
            -plt_wdth / 2,
            plt_wdth / 2,
            -plt_wdth / 2,
            plt_wdth / 2,
        ],
        vmin=np.log10(temp_range[0]),
        vmax=np.log10(temp_range[1]),
        cmap=tempcmap,
    )

    lum_cbar_ax = ax.inset_axes([0.05, 0.05, 0.30, 0.028], alpha=0.8)
    lum_cbar = fig.colorbar(
        f3_lum_image, cax=lum_cbar_ax, pad=0, orientation="horizontal"
    )
    lum_cbar.ax.xaxis.set_tick_params(pad=2)
    lum_cbar.set_label(
        r"$\mathrm{\log\:UV\:SB}$" r"$\:\mathrm{\left(erg\:\:s^{-1}\:pc^{-2}\right)}$",
        labelpad=-15,
    )

    gas_cbar_ax = ax.inset_axes([0.35, 0.05, 0.30, 0.028], alpha=0.8)
    gas_cbar = fig.colorbar(
        cm.ScalarMappable(
            norm=mpl.colors.Normalize(np.log10(gas_range[0]), np.log10(gas_range[1])),
            cmap=gascmap,
        ),
        cax=gas_cbar_ax,
        orientation="horizontal",
        pad=0,
    )
    gas_cbar.ax.xaxis.set_tick_params(pad=2)
    gas_cbar.set_label(
        label=r"$\mathrm{\log\:Gas\:Density\:\left(g \: cm^{-2}\right)}$", labelpad=-15
    )

    temp_cbar_ax = ax.inset_axes([0.65, 0.05, 0.30, 0.028], alpha=0.8)
    temp_cbar = fig.colorbar(
        cm.ScalarMappable(
            norm=mpl.colors.Normalize(np.log10(temp_range[0]), np.log10(temp_range[1])),
            cmap=tempcmap,
        ),
        cax=temp_cbar_ax,
        orientation="horizontal",
        pad=0,
    )
    temp_cbar.ax.xaxis.set_tick_params(pad=2)
    temp_cbar.set_label(label=r"$\mathrm{\log\:Gas\:Temperature\:(K)}$", labelpad=-15)

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
            f3_t_myr, f3_redshift
        ),
        ha="left",
        va="center",
        color="white",
        transform=ax.transAxes,
    )
    # efficiency labels
    ax.text(
        0.95,
        0.95,
        r"$\mathrm{low-SFE\:(35\%)}$",
        ha="right",
        va="center",
        color="white",
        transform=ax.transAxes,
    )

    # save frame
    ax.axis("off")
    output_path = os.path.join(
        sequence_dir,
        "render_{}_{}.png".format(
            outnum_f3, str(np.round(f3_redshift, 3)).replace(".", "_")
        ),
    )
    plt.savefig(
        os.path.expanduser(output_path), dpi=400, bbox_inches="tight", pad_inches=0.00
    )
    # plt.close("all")
    print(">Saved:", output_path)
