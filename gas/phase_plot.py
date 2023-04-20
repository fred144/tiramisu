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

# mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0

cell_fields, epf = ram_fields()
datadir = os.path.relpath("../../cosm_test_data/refine")
datadir = os.path.relpath("../../sim_data/cluster_evolution")

sim_run = datadir.split("/")[-1]

snaps, snap_strings = filter_snapshots(datadir, 150, 1450, sampling=50, str_snaps=True)

simulation_run = datadir

dm_container = os.path.join("../../container_ram-py", "dm_hop", sim_run)
if not os.path.exists(dm_container):
    print("====================================================")
    print("Creating ram-py container", dm_container)
    print("====================================================")
    os.makedirs(dm_container)

container = os.path.join("../../container_ram-py", "gas", sim_run)
if not os.path.exists(container):
    print("====================================================")
    print("Creating ram-py container", container)
    print("====================================================")
    os.makedirs(container)

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

    hc = HaloCatalog(
        data_ds=ds,
        finder_method="hop",
        finder_kwargs={"ptype": "DM", "dm_only": False},
        output_dir=dm_container,
    )
    hc.create()
    hop_catalogue = "{}/info_{}/info_{}.{}.h5".format(
        dm_container,
        snap_strings[i],
        snap_strings[i],
        processor_number,
    )
    #%%
    # need to read in using yt for virial radius for some reason unknown units in catalogue
    cata_yt = yt.load(hop_catalogue)
    cata_yt = cata_yt.all_data()
    dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    haloidx = np.argmax(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
    vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]
    # useful info
    current_time = float(ds.current_time.in_units("Myr"))
    redshft = ds.current_redshift
    t.append(current_time)
    z.append(redshft)
    m_vir.append(dm_halo_m)
    r_vir.append(vir_rad)
    tot_m_star.append(np.sum(np.array(ad["star", "particle_mass"].to("Msun"))))
    # center of the major halo
    halo_x = ds.arr(cata_yt["all", "particle_position_x"]).to("pc")[haloidx]
    halo_y = ds.arr(cata_yt["all", "particle_position_y"]).to("pc")[haloidx]
    halo_z = ds.arr(cata_yt["all", "particle_position_z"]).to("pc")[haloidx]

    galaxy = ds.sphere(
        [halo_x, halo_y, halo_z],
        ds.arr(cata_yt["all", "virial_radius"]).to("pc")[haloidx],
    )
    plot = yt.PhasePlot(
        galaxy,
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "mass"),
        weight_field=None,
    )

    # Set the units of mass to be in solar masses (not the default in cgs)
    plot.set_unit(("gas", "mass"), "Msun")
    plot.save()
    #%%
    profile2d = galaxy.profile(
        # the x bin field, the y bin field
        [("gas", "density"), ("gas", "temperature")],
        [("gas", "mass")],  # the profile field
        weight_field=None,  # sums each quantity in each bin
        n_bins=(132, 132),
    )

    #%%

    gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T
    temp = np.array(profile2d.y)
    dens = np.array(profile2d.x) / 1.6e-24
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=400)
    nt_image = ax.imshow(
        np.log10(gas_mass),
        # np.log10(
        #     gas_mass,
        #     where=(gas_mass != 0),
        #     out=np.full_like(gas_mass, 24),
        # ),
        origin="lower",
        extent=[
            np.log10(dens.min()),
            np.log10(dens.max()),
            np.log10(temp.min()),
            np.log10(temp.max()),
        ],
        cmap="inferno",
        # vmin=24,
        # vmax=39,
    )

    ax.set(
        ylabel=r"$\mathrm{Temperature\:(T)}$",
        xlabel=r"$\mathrm{nH\:(\:cm^{-3})}$",
    )
    ax.text(
        0.05,
        0.05,
        (r"$\mathrm{{t = {:.2f} \: Myr}}$" "\n" r"$\mathrm{{z = {:.2f} }}$").format(
            current_time, redshft
        ),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    cbar_ax = ax.inset_axes([0, 1.02, 1, 0.05])
    bar = fig.colorbar(nt_image, cax=cbar_ax, pad=0, orientation="horizontal")
    # bar .ax.xaxis.set_tick_params(pad=2)
    bar.set_label(r"$\mathrm{\log_{10}\:Mass\:(M_{\odot}})$")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.xaxis.set_ticks_position("top")

    plt.savefig(
        os.path.join(container, f"{snap_strings[i]}.png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.00,
    )
