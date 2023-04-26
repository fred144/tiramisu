import sys

sys.path.insert(1, "/homes/fgarcia4/py-virtual-envs/master/lib/python3.7/site-packages")
sys.path.append("..")  # makes sure that importing the modules work

import yt
import numpy as np
import os
from tools.scanner import filter_snapshots
from tools.ram_fields import ram_fields
import h5py as h5
from yt.funcs import mylog
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import warnings
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0

cell_fields, epf = ram_fields()
# datadir = os.path.relpath("../../cosm_test_data/refine")
datadir = os.path.relpath("../../sim_data/cluster_evolution/fs07_refine")


snaps, snap_strings = filter_snapshots(datadir, 150, 1450, sampling=25, str_snaps=True)
# simulation_run = datadir
plot_name = "nT_metal_phase_sfregion"

lims = {
    ("gas", "density"): ((5e-31, "g/cm**3"), (1e-18, "g/cm**3")),
    ("gas", "temperature"): ((1, "K"), (1e9, "K")),
    ("gas", "mass"): ((1e-6, "msun"), (1e10, "msun")),
}
m_h = 1.6735e-24  # grams
sim_run = datadir.split("/")[-1]
dm_container = os.path.join("../../container_ram-py", "dm_hop", sim_run)
if not os.path.exists(dm_container):
    print("Creating ram-py container", dm_container)
    os.makedirs(dm_container)

gas_container = os.path.join("../../container_ram-py", plot_name, sim_run)
if not os.path.exists(gas_container):
    print("Creating ram-py container", gas_container)
    os.makedirs(gas_container)

#%%

m_vir = []
r_vir = []
tot_m_star = []
t = []
z = []
snap = []

for i, sn in enumerate(snaps):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))
    print("# reading in", infofile)

    ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()

    hop_catalogue = "{}/info_{}/info_{}.{}.h5".format(
        dm_container,
        snap_strings[i],
        snap_strings[i],
        processor_number,
    )

    if os.path.isfile(hop_catalogue) is True:
        # print("file already exists")
        pass
    else:
        hc = HaloCatalog(
            data_ds=ds,
            finder_method="hop",
            finder_kwargs={"ptype": "DM", "dm_only": False},
            output_dir=dm_container,
        )
        hc.create()
    # need to read in using yt for virial radius for some reason unknown units in catalogue
    cata_yt = yt.load(hop_catalogue)
    cata_yt = cata_yt.all_data()
    dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    haloidx = np.argmax(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]
    m_vir.append(dm_halo_m)
    r_vir.append(vir_rad)

    # center of the major halo
    # halo_x = ds.arr(cata_yt["all", "particle_position_x"]).to("pc")[haloidx]
    # halo_y = ds.arr(cata_yt["all", "particle_position_y"]).to("pc")[haloidx]
    # halo_z = ds.arr(cata_yt["all", "particle_position_z"]).to("pc")[haloidx]
    # galaxy = ds.sphere(
    #     [halo_x, halo_y, halo_z],
    #     ds.arr(cata_yt["all", "virial_radius"]).to("pc")[haloidx],
    # )
    # =============================================================================
    mstar = np.sum(np.array(ad["star", "particle_mass"].to("Msun")))
    tot_m_star.append(mstar)
    efficiency = mstar / dm_halo_m

    galaxy = ds.sphere(
        [
            np.mean(ad["star", "particle_position_x"]).to("pc"),
            np.mean(ad["star", "particle_position_y"]).to("pc"),
            np.mean(ad["star", "particle_position_z"]).to("pc"),
        ],
        (250, "pc"),
    )

    # useful info
    current_time = float(ds.current_time.in_units("Myr"))
    redshft = ds.current_redshift
    t.append(current_time)
    z.append(redshft)

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

    #%%

    profile2d = galaxy.profile(
        # the x bin field, the y bin field
        [("gas", "density"), ("gas", "temperature")],
        [("gas", "mass")],  # the profile field
        weight_field=None,  # sums each quantity in each bin
        n_bins=(125, 125),
        extrema=lims,
    )

    #%%

    gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T
    temp = np.array(profile2d.y)
    dens = np.array(profile2d.x)  # / 1.6e-24
    fig, ax = plt.subplots(1, 1, figsize=(6, 7), dpi=300)
    nt_image = ax.imshow(
        np.log10(gas_mass),
        # np.log10(
        #     gas_mass,
        #     where=(gas_mass != 0),
        #     out=np.full_like(gas_mass, 24),
        # ),
        origin="lower",
        extent=[
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
            np.log10(lims[("gas", "temperature")][0][0]),
            np.log10(lims[("gas", "temperature")][1][0]),
        ],
        cmap="magma",
        vmin=np.log10(lims[("gas", "mass")][0][0]),
        vmax=np.log10(1e6),
        aspect=1.3,
    )
    temp_arr = np.geomspace(
        lims[("gas", "temperature")][0][0], lims[("gas", "temperature")][1][0]
    )
    ncr = 4
    n_crit = 5e4 * (temp_arr / 100) * ((1 + redshft) / 10) ** 2 * (ncr / 4) ** -2
    ax.fill_between(np.log10(n_crit), np.log10(temp_arr), color="grey")

    ax.set(
        ylabel=r"$\log_{10}\:{\rm Temperature\:(K)}$",
        xlabel=r"$\log_{10}\:n_{\rm H} \:{\rm (cm^{-3}})$",
        xlim=(
            np.log10(lims[("gas", "density")][0][0] / m_h),
            np.log10(lims[("gas", "density")][1][0] / m_h),
        ),
        ylim=(
            np.log10(lims[("gas", "temperature")][0][0]),
            np.log10(lims[("gas", "temperature")][1][0]),
        ),
    )
    ax.text(
        0.65,
        0.95,
        (
            r"$\mathrm{{t = {:.2f} \: Myr}}$"
            "\n"
            r"$\mathrm{{z = {:.2f} }}$"
            "\n"
            r"$\mathrm{{SFE_{{halo}} = {:.2e}}}$"
            "\n"
            r"${{\rm R_{{ SF \, region}}}} = 250 {{\rm pc}}$"
            "\n"
            r"${{\rm run-}}\mathrm{{{}}}$"
        ).format(current_time, redshft, efficiency, sim_run),
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.95,
        0.08,
        r"$n_{ \rm crit}$",
        ha="right",
        va="bottom",
        color="white",
        transform=ax.transAxes,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    cbar_ax = ax.inset_axes([0, 1.02, 1, 0.05])
    bar = fig.colorbar(nt_image, cax=cbar_ax, pad=0, orientation="horizontal")
    # bar .ax.xaxis.set_tick_params(pad=2)
    bar.set_label(r"$\mathrm{\log_{10}\:Total\:Mass\:(M_{\odot}})$")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.xaxis.set_ticks_position("top")

    save_path = os.path.join(
        gas_container,
        r"{}-{}.png".format(
            # plot_name,
            snap_strings[i],
            "{:.2f}".format(current_time).replace(".", "_"),
        ),
    )
    print("Saved", save_path)
    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
        # pad_inches=0.00,
    )
