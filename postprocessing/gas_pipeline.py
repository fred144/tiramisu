"""
Metallicity as a function of density, with mass bins
Divided into three different gas phases
Region centered on star formation, radius of 0.5 kpc
"""

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
import cmasher as cmr

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0
cell_fields, epf = ram_fields()


if __name__ == "__main__":
    m_h = 1.6735e-24  # grams
    r_sf = 500  # radii for sf in pc
    zsun = 0.02
    n_crit = 5e4
    cold_phase_t = (0, 100)
    warm_phase_t = (100, 5e4)
    hot_phase = (5.001e4, 1e9)
    temp_cuts = [cold_phase_t, warm_phase_t, hot_phase]
    tlabels = [
        r"CNM ($T < 100$ K)",
        r"WNM  ($ 100 < T \leq 5 \times 10^4$ K)",
        r"Hot ($T > 5 \times 10^4$ K)",
    ]
    lims = {
        ("gas", "density"): ((5e-30, "g/cm**3"), (1e-18, "g/cm**3")),
        ("ramses", "Metallicity"): (1e-6 * zsun, 5 * zsun),
        ("gas", "mass"): ((1e-2, "msun"), (1e6, "msun")),
    }

    if len(sys.argv) != 5:
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
    logsfc_path = os.path.join(sys.argv[1], "logSFC")
    start_snapshot = int(sys.argv[2])
    end_snapshot = int(sys.argv[3])
    step = int(sys.argv[4])

    sim_run = os.path.basename(os.path.normpath(datadir))
    fpaths, snums = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )

    # first starburst in the CC-fid run
    # queiscent phase after 1st star burst, before 2nd snap 203 - 370
    # second starburst in the CC-fid run snap 377 - 389
    # after the second starburst snap 402 - 432

    # =============================================================================

    # datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    # logsfc_path = os.path.expanduser(os.path.join(datadir, "logSFC"))

    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     370,
    #     390,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # render_nickname = "test"

    # =============================================================================

    # run save
    sim_run = os.path.basename(os.path.normpath(datadir))
    postprocessed_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "post_processed",
        "gas_properties",
        sim_run,
    )
    dm_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "dm_hop", sim_run
    )
    check_path(dm_container)
    check_path(postprocessed_container)

    cold = []
    warm = []
    hot = []

    myrs = []
    redshifts = []

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
        myrs.append(t_myr)
        redshifts.append(redshift)

        hop_catalogue = "{}/info_{}/info_{}.{}.h5".format(
            dm_container,
            snums[i],
            snums[i],
            processor_number,
        )

        if os.path.isfile(hop_catalogue) is True:
            print(">> catalogue already exists")
            pass
        else:
            hc = HaloCatalog(
                data_ds=ds,
                finder_method="hop",
                finder_kwargs={"ptype": "DM", "dm_only": False},
                output_dir=dm_container,
            )
            hc.create()

        # need to read in using yt for virial radius for
        # some reason unknown units in catalogue
        cata_yt = yt.load(hop_catalogue)
        cata_yt = cata_yt.all_data()
        dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
        haloidx = np.argmax(
            np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun"))
        )  # most massive halo is the central halo
        vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]
        dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
        print("virial radius [pc]", vir_rad)

        x_pos = ad["star", "particle_position_x"]
        y_pos = ad["star", "particle_position_y"]
        z_pos = ad["star", "particle_position_z"]
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)

        galaxy_center = np.array([x_center, y_center, z_center])

        galaxy_radius = 500  # pc
        mass_hyrogen = 1.6735e-24  # grams
        end_region = 1e4  # "pc"
        bins = np.geomspace(1, end_region, 60, endpoint=True)

        # galaxy properties
        sf_region = ds.sphere(galaxy_center, (galaxy_radius, "pc"))
        full_region = ds.sphere(galaxy_center, (end_region, "pc"))

        # let's calculate the different phases by mass in the galaxy or ISM
        cnm = sf_region.exclude_outside(("gas", "temperature"), 0, 100)
        cnm_mass = (
            cnm.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun").value
        )

        wnm = sf_region.exclude_outside(("gas", "temperature"), 100.001, 5.001e4)
        wnm_mass = (
            wnm.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun").value
        )

        hot = sf_region.exclude_below(("gas", "temperature"), 5.001e4)
        hot_mass = (
            hot.quantities.total_quantity(("gas", "cell_mass")).in_units("Msun").value
        )

        # virial radius and outside, "IGM"
        igm = full_region.exclude_below(("index", "radius"), vir_rad, units="pc")
        igm_mean_Z = igm.mean(("ramses", "Metallicity")) / zsun
        igm_mean_Z_weighted = igm.quantities["WeightedAverageQuantity"](
            ("ramses", "Metallicity"), weight=("gas", "mass")
        )
        igm_mean_Z_weighted /= zsun

        # within virial radius, but outside of the galaxy
        cgm = full_region.exclude_outside(
            ("index", "radius"), galaxy_radius, vir_rad, units="pc"
        )
        cgm_mean_Z = cgm.mean(("ramses", "Metallicity")) / zsun
        cgm_mean_Z_weighted = cgm.quantities["WeightedAverageQuantity"](
            ("ramses", "Metallicity"), weight=("gas", "mass")
        )
        cgm_mean_Z_weighted /= zsun

        sf_region_mean_Z = sf_region.mean(("ramses", "Metallicity")) / zsun
        sf_region_mean_Z_weighted = sf_region.quantities["WeightedAverageQuantity"](
            ("ramses", "Metallicity"), weight=("gas", "mass")
        )
        sf_region_mean_Z_weighted /= zsun

        # let's construct the radial velocity profile for the halo
        sp0 = ds.sphere(galaxy_center, (end_region, "pc"))
        # Compute the bulk velocity from the cells in this sphere
        bulk_vel = sp0.quantities.bulk_velocity()
        # Get the second sphere
        sp1 = ds.sphere(galaxy_center, (end_region, "pc"))

        # Set the bulk velocity field parameter
        sp1.set_field_parameter("bulk_velocity", bulk_vel)

        # Radial profile with correction for bulk velocity
        radvel = yt.create_profile(
            sp1,
            ("index", "radius"),
            ("gas", "radial_velocity"),
            units={("index", "radius"): "pc"},
            override_bins={("index", "radius"): (bins, "pc")},
        )

        density = yt.create_profile(
            sp1,
            ("index", "radius"),
            ("gas", "density"),
            weight_field=("gas", "density"),
            units={("index", "radius"): "pc", ("gas", "density"): "g/cm**3"},
            override_bins={("index", "radius"): (bins, "pc")},
        )

        metallicity = yt.create_profile(
            sp1,
            ("index", "radius"),
            ("ramses", "Metallicity"),
            weight_field=("gas", "density"),  #!!! try mass weight
            units={("index", "radius"): "pc"},
            override_bins={("index", "radius"): (bins, "pc")},
        )

        temperature = yt.create_profile(
            sp1,
            ("index", "radius"),
            ("gas", "temperature"),
            weight_field=("gas", "density"),
            units={("index", "radius"): "pc"},
            override_bins={("index", "radius"): (bins, "pc")},
        )

        radvel_profile = radvel[("gas", "radial_velocity")].in_units("km/s").value
        density_profile = (
            density[("gas", "density")].in_units("g/cm**3").value / mass_hyrogen
        )
        metallicity_profile = metallicity[("ramses", "Metallicity")] / zsun
        temperature_profile = temperature[("gas", "temperature")].in_units("K").value

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(
        #     radvel.x.value,
        #     radvel[("gas", "radial_velocity")].in_units("km/s").value,
        # )
        # ax.scatter(
        #     density.x.value,
        #     density[("gas", "density")].in_units("g/cm**3").value / mass_hyrogen,
        # )
        # ax.plot(
        #     metallicity.x.value,
        #     metallicity[("ramses", "Metallicity")] / zsun,
        # )
        # ax.plot(
        #     temperature.x.value,
        #     temperature[("gas", "temperature")].in_units("K").value,
        # )

        # ax.set_xlabel(r"$\mathrm{r\ (pc)}$")
        # # ax.set_ylabel(r"$\mathrm{v_r\ (km/s)}$")
        # ax.legend(["{:.2f}".format(t_myr)])
        # ax.set_xscale("log")
        # # ax.set_ylim(1e-3, 1)
        # # # ax.set_xlim(1e-5, 2)
        # ax.set_yscale("log")
        # plt.show()

        save_time = "{:.2f}".format(t_myr).replace(".", "_")
        save_redshift = "{:.3f}".format(redshift).replace(".", "_")

        f = h5.File(
            os.path.join(
                postprocessed_container,
                "profiles-{}-{}-myr-z-{}.hdf5".format(
                    snums[i], save_time, save_redshift
                ),
            ),
            "w",
        )
        f.create_dataset("Header/redshift", data=redshift, dtype="f")
        f.create_dataset("Header/time", data=t_myr, dtype="f")
        f.create_dataset("Halo/radius", data=vir_rad, dtype="f")
        f.create_dataset("Halo/Mass", data=dm_halo_m, dtype="f")

        f.create_dataset("Galaxy/MeanMetallicity", data=sf_region_mean_Z, dtype="f")
        f.create_dataset(
            "Galaxy/MeanWeightedMetallicity", data=sf_region_mean_Z_weighted, dtype="f"
        )

        f.create_dataset("CGM/MeanMetallicity", data=cgm_mean_Z, dtype="f")
        f.create_dataset(
            "CGM/MeanWeightedMetallicity", data=cgm_mean_Z_weighted, dtype="f"
        )

        f.create_dataset("IGM/MeanMetallicity", data=igm_mean_Z, dtype="f")
        f.create_dataset(
            "IGM/MeanWeightedMetallicity", data=igm_mean_Z_weighted, dtype="f"
        )

        f.create_dataset("Galaxy/ColdNeutralMediumMass", data=cnm_mass, dtype="f")
        f.create_dataset("Galaxy/WarmNeutralMediumMass", data=wnm_mass, dtype="f")
        f.create_dataset("Galaxy/HotGasMass", data=hot_mass, dtype="f")

        f.create_dataset("Profiles/Radius", data=bins, dtype="f")
        f.create_dataset("Profiles/RadialVelocity", data=radvel_profile, dtype="f")
        f.create_dataset("Profiles/Density", data=density_profile, dtype="f")
        f.create_dataset(
            "Profiles/MetalDensityWeighted", data=metallicity_profile, dtype="f"
        )
        f.create_dataset(
            "Profiles/TempDensityWeighted", data=temperature_profile, dtype="f"
        )
        f.close()
