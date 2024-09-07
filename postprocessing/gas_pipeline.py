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
from yt.fields.api import ValidateParameter

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
processor_number = 0
cell_fields, epf = ram_fields()


def _my_radial_velocity(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity").in_units("km/s")
    else:
        bv = data.ds.arr(np.zeros(3), "km/s")

    xv = data["gas", "velocity_x"] - bv[0]
    yv = data["gas", "velocity_y"] - bv[1]
    zv = data["gas", "velocity_z"] - bv[2]
    # what is supplied as center to the plotting routine
    center = data.get_field_parameter("center")
    x_hat = data["gas", "x"] - center[0]
    y_hat = data["gas", "y"] - center[1]
    z_hat = data["gas", "z"] - center[2]
    r = np.sqrt(x_hat * x_hat + y_hat * y_hat + z_hat * z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return xv * x_hat + yv * y_hat + zv * z_hat


yt.add_field(
    ("gas", "my_radial_velocity"),
    function=_my_radial_velocity,
    sampling_type="cell",
    units="km/s",
    take_log=False,
    validators=[
        ValidateParameter(["center", "bulk_velocity"]),
    ],
)

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

    # if len(sys.argv) != 5:
    #     print(sys.argv[0], "usage:")
    #     print("{} snapshot_dir start_snap end_snap step ".format(sys.argv[0]))
    #     exit()
    # else:
    #     print("********************************************************************")
    #     print("post processing global gas quantitites")
    #     print("********************************************************************")

    # datadir = sys.argv[1]
    # logsfc_path = os.path.join(sys.argv[1], "logSFC")
    # start_snapshot = int(sys.argv[2])
    # end_snapshot = int(sys.argv[3])
    # step = int(sys.argv[4])

    # sim_run = os.path.basename(os.path.normpath(datadir))
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     start_snapshot,
    #     end_snapshot,
    #     sampling=step,
    #     str_snaps=True,
    # )

    # first starburst in the CC-fid run
    # queiscent phase after 1st star burst, before 2nd snap 203 - 370
    # second starburst in the CC-fid run snap 377 - 389
    # after the second starburst snap 402 - 432

    # =============================================================================

    datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    logsfc_path = os.path.expanduser(os.path.join(datadir, "logSFC"))

    fpaths, snums = filter_snapshots(
        datadir,
        153,
        304,
        sampling=2,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

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
        try:
            ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
            ad = ds.all_data()
        except:
            print("having trouble making regon cuts, skipping")
            continue

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
            print(hop_catalogue)
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
        cata_yt = h5.File(hop_catalogue, "r")
        # cata_yt = yt.load(hop_catalogue)
        # cata_yt = cata_yt.all_data()

        dm_halo_m = np.max(np.array(cata_yt["particle_mass"][:]))  # Msun
        haloidx = np.argmax(dm_halo_m)  # most massive halo is the central halo

        # kpc is the default
        vir_rad = np.array(ds.arr(cata_yt["virial_radius"][:], "kpc").to("pc"))[haloidx]
        x_pos = np.array(
            ds.arr(cata_yt["particle_position_x"][:], "code_length").to("pc")
        )[haloidx]

        y_pos = np.array(
            ds.arr(cata_yt["particle_position_y"][:], "code_length").to("pc")
        )[haloidx]
        z_pos = np.array(
            ds.arr(cata_yt["particle_position_y"][:], "code_length").to("pc")
        )[haloidx]

        halo_center = ds.arr(np.array([x_pos, y_pos, z_pos]), "pc")

        print(
            "virial radius [pc]",
            vir_rad,
            ", virial mass {:.2e} [Msun]".format(dm_halo_m),
        )

        x_pos = ad["star", "particle_position_x"]
        y_pos = ad["star", "particle_position_y"]
        z_pos = ad["star", "particle_position_z"]
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)

        cata_yt.close()

        # centroid of stars
        galaxy_center = ds.arr(
            np.array([x_center, y_center, z_center]), "code_length"
        ).in_units("pc")
        # galaxy_radius = 500  # pc
        galaxy_radius = 0.1 * vir_rad
        mass_hyrogen = 1.6735e-24  # grams
        end_region = 1e4  # "pc"
        bins = np.geomspace(1, end_region, 60, endpoint=True)

        # galaxy properties
        sf_region = ds.sphere(galaxy_center, (galaxy_radius, "pc"))
        full_region = ds.sphere(galaxy_center, (end_region, "pc"))  # 10 kpc region
        vir_region = ds.sphere(galaxy_center, (vir_rad, "pc"))  # vir radius, usual 2kpc

        #                       make region cuts
        try:
            shell_thicknes = ds.arr(0.1 * galaxy_radius, "pc")
            print(
                "ISM outflow rates computed b/w [pc] r = ",
                galaxy_radius,
                galaxy_radius + shell_thicknes.value,
            )
            # measure SF region
            inner_spherical_shell = full_region.exclude_outside(
                ("index", "radius"),
                galaxy_radius,
                galaxy_radius + shell_thicknes.value,
                units="pc",
            )
            vrads = inner_spherical_shell["radial_velocity"].to("km/s")
            outwards_mask = vrads > 0
            v_out = vrads[outwards_mask]
            inwards_mask = vrads < 0
            v_in = vrads[inwards_mask]
            # mass of each gas cell in the shell
            mass_of_gas = inner_spherical_shell["gas", "cell_mass"].in_units("Msun")
            # metal abundance in the shells
            shell_metal = inner_spherical_shell["ramses", "Metallicity"] / zsun
            # mass in metals in the shells
            mass_of_metals = mass_of_gas * shell_metal

            ########## measure Virial Radius interface
            outer_shell_thicknes = ds.arr(vir_rad * 0.1, "pc")
            print(
                "halo outflow rates computed b/w [pc] r = ",
                vir_rad,
                vir_rad + outer_shell_thicknes.value,
            )
            outer_spherical_shell = full_region.exclude_outside(
                ("index", "radius"),
                vir_rad,
                vir_rad + outer_shell_thicknes.value,
                units="pc",
            )
            outer_vrads = outer_spherical_shell["radial_velocity"].to("km/s")
            halo_outwards_mask = outer_vrads > 0
            halo_v_out = outer_vrads[halo_outwards_mask]
            halo_inwards_mask = outer_vrads < 0
            halo_v_in = outer_vrads[halo_inwards_mask]
            # mass of each gas cell in the shell
            halo_mass_of_gas = outer_spherical_shell["gas", "cell_mass"].in_units(
                "Msun"
            )
            # metal abundance in the shells
            halo_shell_metal = outer_spherical_shell["ramses", "Metallicity"] / zsun
            # mass in metals in the shells
            halo_mass_of_metals = halo_mass_of_gas * halo_shell_metal
        except:
            print("having trouble making regon cuts, skipping")
            continue

        # metals going out
        metal_out = mass_of_metals[outwards_mask]
        dmetalout_dt = (metal_out * v_out).sum() / shell_thicknes.to("km")
        metalmass_out_per_year = dmetalout_dt.to("Msun/yr")
        # total gas mass going out
        gasmass_out = (mass_of_gas[outwards_mask] * v_out).sum() / shell_thicknes.to(
            "km"
        )
        gasmass_per_year = gasmass_out.to("Msun/yr")

        # metals inflow
        metal_in = mass_of_metals[inwards_mask]
        dmetalin_dt = (metal_in * v_in).sum() / shell_thicknes.to("km")
        metalmass_in_per_year = dmetalin_dt.to("Msun/yr")
        # total gas mass going out
        gasmass_in = (mass_of_gas[inwards_mask] * v_in).sum() / shell_thicknes.to("km")
        gasmass_inflow_per_year = gasmass_in.to("Msun/yr")
        print("mass outflow rate [msun / yr]", gasmass_per_year)
        print("mass inflow rate [msun / yr]", -gasmass_inflow_per_year)

        ######### now for the Halo interface

        # metals outflow
        halo_metal_out = halo_mass_of_metals[halo_outwards_mask]
        halo_dmetalout_dt = (
            halo_metal_out * halo_v_out
        ).sum() / outer_shell_thicknes.to("km")
        halo_metalmass_out_per_year = halo_dmetalout_dt.to("Msun/yr")

        # total gas mass going out
        halo_gasmass_out = (
            halo_mass_of_gas[halo_outwards_mask] * halo_v_out
        ).sum() / outer_shell_thicknes.to("km")
        halo_gasmass_per_year = halo_gasmass_out.to("Msun/yr")

        # metals inflow
        halo_metal_in = halo_mass_of_metals[halo_inwards_mask]
        halo_dmetalin_dt = (halo_metal_in * halo_v_in).sum() / outer_shell_thicknes.to(
            "km"
        )
        halo_metalmass_in_per_year = halo_dmetalin_dt.to("Msun/yr")

        # total gas mass going out
        halo_gasmass_in = (
            halo_mass_of_gas[halo_inwards_mask] * halo_v_in
        ).sum() / outer_shell_thicknes.to("km")
        halo_gasmass_inflow_per_year = halo_gasmass_in.to("Msun/yr")

        print("halo mass outflow rate [msun / yr]", halo_gasmass_per_year)
        print("halo mass inflow rate [msun / yr]", -halo_gasmass_inflow_per_year)

        # within virial radius, but outside of the galaxy
        cgm = full_region.exclude_outside(
            ("index", "radius"), galaxy_radius, vir_rad, units="pc"
        )

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

        # mean vir metallicity

        vir_Mgas = vir_region["gas", "cell_mass"].in_units("Msun").sum().to_value()
        # for each cell, what is the mass
        vir_Mgas_cell = vir_region["gas", "cell_mass"].in_units("Msun").to_value()
        vir_Zmetal_cell = vir_region["ramses", "Metallicity"].to_value() / zsun
        # for each cell, what is the Mmetal/Mgas in each cell
        vir_Mmetal_cell = vir_Mgas_cell * vir_Zmetal_cell  # metal mass in Msun
        vir_mean_Z = vir_Mmetal_cell.sum() / vir_Mgas  # average metallicity

        # virial radius and outside, "IGM"
        igm = full_region.exclude_below(("index", "radius"), vir_rad, units="pc")
        # sum of gas in region
        igm_Mgas = igm["gas", "cell_mass"].in_units("Msun").sum().to_value()
        # for each cell, what is the mass
        igm_Mgas_cell = igm["gas", "cell_mass"].in_units("Msun").to_value()
        # for each cell, what is the Mmetal/Mgas in each cell
        igm_Zmetal_cell = igm["ramses", "Metallicity"].to_value() / zsun
        igm_Mmetal_cell = igm_Mgas_cell * igm_Zmetal_cell  # metal mass in Msun
        igm_mean_Z = igm_Mmetal_cell.sum() / igm_Mgas  # average metallicity

        cgm_Mgas = cgm["gas", "cell_mass"].in_units("Msun").sum().to_value()
        cgm_Mgas_cell = cgm["gas", "cell_mass"].in_units("Msun").to_value()
        cgm_Zmetal_cell = cgm["ramses", "Metallicity"].to_value() / zsun
        cgm_Mmetal_cell = cgm_Mgas_cell * cgm_Zmetal_cell  # metal mass in Msun
        cgm_mean_Z = cgm_Mmetal_cell.sum() / cgm_Mgas  # average metallicity

        sf_Mgas = sf_region["gas", "cell_mass"].in_units("Msun").sum().to_value()
        sf_Mgas_cell = sf_region["gas", "cell_mass"].in_units("Msun").to_value()
        sf_Zmetal_cell = sf_region["ramses", "Metallicity"].to_value() / zsun
        sf_Mmetal_cell = sf_Mgas_cell * sf_Zmetal_cell  # metal mass in Msun
        sf_mean_Z = sf_Mmetal_cell.sum() / sf_Mgas  # average metallicity

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

        # gas properties
        f.create_dataset(
            "Winds/MassOutFlowRate", data=gasmass_per_year.value, dtype="f"
        )
        f.create_dataset(
            "Winds/MetalMassOutFlowRate", data=metalmass_out_per_year.value, dtype="f"
        )

        f.create_dataset(
            "Winds/MassInFlowRate", data=gasmass_inflow_per_year.value, dtype="f"
        )
        f.create_dataset(
            "Winds/MetalMassInFlowRate", data=metalmass_in_per_year.value, dtype="f"
        )

        ## same but for Halo
        f.create_dataset(
            "HaloWinds/MassOutFlowRate", data=halo_gasmass_per_year.value, dtype="f"
        )
        f.create_dataset(
            "HaloWinds/MetalMassOutFlowRate",
            data=halo_metalmass_out_per_year.value,
            dtype="f",
        )

        f.create_dataset(
            "HaloWinds/MassInFlowRate",
            data=halo_gasmass_inflow_per_year.value,
            dtype="f",
        )
        f.create_dataset(
            "HaloWinds/MetalMassInFlowRate",
            data=halo_metalmass_in_per_year.value,
            dtype="f",
        )

        f.create_dataset("Halo/radius", data=vir_rad, dtype="f")
        f.create_dataset("Halo/Mass", data=dm_halo_m, dtype="f")
        f.create_dataset("Halo/MeanMetallicity", data=vir_mean_Z, dtype="f")
        f.create_dataset("Halo/MetalMass", data=vir_Mmetal_cell.sum(), dtype="f")

        f.create_dataset("Galaxy/GasMass", data=sf_Mgas, dtype="f")
        f.create_dataset("Galaxy/ColdNeutralMediumMass", data=cnm_mass, dtype="f")
        f.create_dataset("Galaxy/WarmNeutralMediumMass", data=wnm_mass, dtype="f")
        f.create_dataset("Galaxy/HotGasMass", data=hot_mass, dtype="f")
        f.create_dataset("Galaxy/MeanMetallicity", data=sf_mean_Z, dtype="f")
        f.create_dataset("Galaxy/MetalMass", data=sf_Mmetal_cell.sum(), dtype="f")

        f.create_dataset("CGM/MeanMetallicity", data=cgm_mean_Z, dtype="f")
        f.create_dataset("CGM/MetalMass", data=cgm_Mmetal_cell.sum(), dtype="f")
        f.create_dataset("CGM/GasMass", data=cgm_Mgas, dtype="f")

        f.create_dataset("IGM/MeanMetallicity", data=igm_mean_Z, dtype="f")
        f.create_dataset("IGM/MetalMass", data=igm_Mmetal_cell.sum(), dtype="f")
        f.create_dataset("IGM/GasMass", data=igm_Mgas, dtype="f")

        f.create_dataset("Profiles/Radius", data=bins, dtype="f")
        f.create_dataset("Profiles/RadialVelocity", data=radvel_profile, dtype="f")
        f.create_dataset("Profiles/Density", data=density_profile, dtype="f")
        f.create_dataset(
            "Profiles/MetalDensityWeighted", data=metallicity_profile, dtype="f"
        )
        f.create_dataset(
            "Profiles/TempDensityWeighted", data=temperature_profile, dtype="f"
        )
        print(
            "saved",
            os.path.join(
                postprocessed_container,
                "profiles-{}-{}-myr-z-{}.hdf5".format(
                    snums[i], save_time, save_redshift
                ),
            ),
        )
        f.close()
