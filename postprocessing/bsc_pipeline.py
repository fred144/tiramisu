import sys

sys.path.append("..")  # makes sure that importing the modules work
import numpy as np
import os
import h5py as h5
from src.lum.lum_lookup import lum_look_up_table
from tools.cosmo import code_age_to_myr
from tools.ram_fields import ram_fields
from tools.fscanner import filter_snapshots
import glob
from scipy import stats as st
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog


def check_path(path):
    if not os.path.exists(path):
        print("Creating container", path)
        os.makedirs(path)
    else:
        pass


def modified_king_model(
    radius: float, central_projden: float, core_rad: float, alpha: float, bg: float
):
    """
    Parameters
    ----------
    radius : float
        array of the radial distance from center of the BSC in pc.
    central_projden : float
        central mass density fitting parameter, msun/pc^2
    core_rad : float
        core radius in pc.
    alpha : float
        alpha parameter.
    bg : float
        background surface density.

    Returns
    -------
    sigma : float
        array of the smooth surface density profile

    """
    sigma = bg + (central_projden / (1 + (radius / core_rad) ** alpha))
    return sigma


def trunc_radius(sigma_0: float, r_c: float, alpha: float, sigma_bg: float):
    """
    get the truncation radius.
    calculated by setting to
    1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)

    Parameters
    ----------
    sigma_0 : float
        fitted central density.  msun/pc^2
    r_c : float
        fitted core radius in pc
    alpha : float
        alpha parameter
    sigma_bg : float
        fitted_bg
    Returns
    -------
    r_trunc : float
        truncation radius for the cluster

    """
    # can change 1.5 to any vale. Here the truncation radius is when the baground
    # projected density is roughly half of the clusted projected density.
    r_trunc = (r_c**alpha * ((sigma_0 / ((1.5 - 1) * sigma_bg) - 1))) ** (1 / alpha)
    return r_trunc


def projected_surf_densities(
    x_coord: float,
    y_coord: float,
    lums: float,
    masses: float,
    radius: float,
    num_bins: int = 25,
    log_bins=True,
    calc_half_r=True,
    dr=None,
):
    """
    Gets projected density profiles surf_density
    Log bins by default.
    Assumes that the cluster fed to this has already been translated to (0,0)

    Parameters
    ----------
    x_coord : float
        x coordinate of the stars in the projected coords .
    y_coord : float
        y coordinate of the stars in the projected coords .
    lums : float
        unloged luminosities of the stars.
    masses : float
        masses of the stars.
    radius : float
        boundary radius as determined by the clump/halo finder.
    num_bins : int, optional
        number of bins in the profile. The default is 25.
    log_bins : TYPE, optional
        log-space bin or not. The default is True.
    calc_half_r : TYPE, optional
        calculate half mass and half light radius. The default is True.
    dr : TYPE, optional
        for linear space, spacing. The default is None.

    Returns
    -------
    list
        bin_ctrs,
        surf_mass_density,
        err_surf_mass_density,
        half_mass_r,
        half_light_r,
        total_clust_m,
        total_clust_lum,

    """

    # TODO: calculate half mass
    starting_point = 0.01  # pc might have to tweak this.

    # stack two-1d arrays
    all_positions = np.vstack((x_coord, y_coord)).T

    if log_bins is True:
        r = np.geomspace(starting_point, radius, num=num_bins, endpoint=True)
        # r_inner = np.geomspace(0, radius, num=num_bins, endpoint=False)
    else:
        r = np.arange(0, radius + dr, dr)
    # print(r)
    distances = np.sqrt(np.sum(np.square(all_positions), axis=1))

    mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=masses)

    lum_per_bin, _ = np.histogram(distances, bins=r, weights=lums)
    count_per_bin, _ = np.histogram(distances, bins=r)

    # mask zero count bins
    mask = count_per_bin > 0
    count_per_bin = count_per_bin[mask]
    mass_per_bin = mass_per_bin[mask]
    lum_per_bin = lum_per_bin[mask]

    # getting bin properties
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)[mask]
    ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]

    # calculate densities
    surf_mass_density = mass_per_bin / ring_areas
    surf_lum_density = lum_per_bin / ring_areas
    surf_number_density = count_per_bin / ring_areas

    # characterize what the typical mass is for a bin
    avg_star_masses = mass_per_bin / count_per_bin
    # piosson error in the surface density
    err_surf_mass_density = np.sqrt(count_per_bin) * (avg_star_masses / ring_areas)
    # sum the bins to get total mass out to the specified cluster radii
    total_clust_m = np.sum(mass_per_bin)
    total_clust_lum = np.sum(lum_per_bin)

    if calc_half_r is True:
        # reevaluate with increased bin resolition to get as close to the actual values
        dr = 0.1
        high_res_r = np.arange(0, radius + 0.01, 0.01)
        high_res_m_per_bin, _ = np.histogram(distances, bins=high_res_r, weights=masses)
        high_res_l_per_bin, _ = np.histogram(distances, bins=high_res_r, weights=lums)
        integrated_mass = np.cumsum(high_res_m_per_bin)
        integrated_light = np.cumsum(high_res_l_per_bin)
        half_mass_point = np.abs(integrated_mass - 0.5 * total_clust_m).argmin()
        half_light_point = np.abs(integrated_light - 0.5 * total_clust_lum).argmin()

        half_mass_r = high_res_r[half_mass_point]
        half_light_r = high_res_r[half_light_point]

        return [
            bin_ctrs,
            surf_mass_density,
            err_surf_mass_density,
            half_mass_r,
            half_light_r,
            total_clust_m,
            total_clust_lum,
        ]
    else:
        return [
            bin_ctrs,
            surf_mass_density,
            err_surf_mass_density,
            total_clust_m,
            total_clust_lum,
        ]


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(sys.argv[0], "usage:")
        print(
            "{} data_directory_to_postprocess start_snapshot end_snapshot step".format(
                sys.argv[0]
            )
        )
        exit()
    else:
        print("********************************************************************")
        print(" running BSC finder / star clump finder")
        print("********************************************************************")

    datadir = sys.argv[1]
    start_snapshot = int(sys.argv[2])
    end_snapshot = int(sys.argv[3])
    step = int(sys.argv[4])

    # local path for test
    # datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")
    # datadir = os.path.relpath("../../garcia23_testdata/fs07_refine")
    # start_snapshot = 500
    # end_snapshot = 500
    # step = 1

    sim_run = os.path.basename(os.path.normpath(datadir))

    snaps, snap_strings = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )

    bsc_cat_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "post_processed",
        "bsc_catalogues",
        sim_run,
    )
    pop2_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "pop2", sim_run
    )

    check_path(bsc_cat_container)

    processor_number = 0
    cell_fields, epf = ram_fields()

    tseries_header = (
        "snapshot time[myr] redshift clumped_mass[msun] "
        "bsc_mass[msun] disrupted_mass[msun] unbound_mass[msun] "
        "clumped_lum UV (150nm) [erg/s]"
        "total_lum UV (150nm) [erg/s] "
        "disrupted_lum UV (150nm) [erg/s] "
        "unbound_lum UV (150nm) [erg/s] "
    )
    time_series_data = []

    for i, sn in enumerate(snaps):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))

        print("# reading in", infofile)

        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        clump_cata_yt = f"{bsc_cat_container}/info_{snap_strings[i]}/info_{snap_strings[i]}.{processor_number}.h5"
        snapshot_container = os.path.join(bsc_cat_container, f"info_{snap_strings[i]}")
        if os.path.isfile(clump_cata_yt) is True:
            print("# file already exists")
            print("# reading in", clump_cata_yt)

            pass
        else:
            hc = HaloCatalog(
                data_ds=ds,
                finder_method="fof",
                finder_kwargs={
                    "ptype": "star",
                    "padding": 0.0001,
                    "link": 0.00001,  # "best"
                    "dm_only": False,
                },
                output_dir=bsc_cat_container,
            )
            hc.create()

        # need to read in using yt for clump radius
        cata_yt = yt.load(clump_cata_yt)
        # make a halo catalogue for yt overplot
        halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
        halo_cat_plotting.load()
        cata_yt = cata_yt.all_data()

        # post processed pop2
        pop2_header = (
            "ID"
            "|CurrentAges[Myr]|"
            " "
            "|log10UV(150nm)Lum[erg/s]|"
            " "
            "|X[pc]"
            "|Y[pc]|"
            "|Z[pc]|"
            " "
            "|Vx[km/s]"
            "|Vy[km/s]|"
            "|Vz[km/s]|"
            " "
            "|mass[Msun]"
        )
        pop2_path = glob.glob(
            os.path.join(pop2_container, "pop2-{}-*".format(snap_strings[i]))
        )[0]

        pop2_data = np.loadtxt(pop2_path)
        window_ctr_pc = pop2_data[5:8, 0]  # pc centered based on star CoM
        ctr_at_code_length = pop2_data[2:5, 0]
        star_ids = pop2_data[:, 1]
        pop2_ages = pop2_data[:, 2]  # in myr
        pop2_lums = pop2_data[:, 3]
        pop2_x = pop2_data[:, 4]
        pop2_y = pop2_data[:, 5]
        pop2_z = pop2_data[:, 6]
        pop2_vx = pop2_data[:, 7]
        pop2_vy = pop2_data[:, 8]
        pop2_vz = pop2_data[:, 9]
        pop2_masses = pop2_data[:, 10]

        # get the clump centers
        cata_h5 = h5.File(clump_cata_yt, "r")
        clump_id = np.array(cata_h5["particle_identifier"])

        x = np.array(cata_h5["particle_position_x"])
        y = np.array(cata_h5["particle_position_y"])
        z = np.array(cata_h5["particle_position_z"])
        x = np.array(ds.arr(x, "code_length").to("pc")) - window_ctr_pc[0]
        y = np.array(ds.arr(y, "code_length").to("pc")) - window_ctr_pc[1]
        z = np.array(ds.arr(z, "code_length").to("pc")) - window_ctr_pc[2]

        # get clump stats
        clump_rad = np.array(ds.arr(cata_yt["all", "virial_radius"], "cm").to("pc"))

        # np.savetxt(bsc_cat_container, X=cat_pc.T, header=header)

        # get particles belonging to each halo
        num_stars_in_clump = np.array(cata_h5["particle_number"])
        start_of_new_clump = np.array(cata_h5["particle_index_start"])
        bound_star_ids = np.array(cata_h5["particles/ids"])
        cata_h5.close()

        reprocessed_clump_cata = []

        # =============================================================================
        #         profiling the clumps
        #         profiled clumps are BSCs
        #         bounds stars are any stars belonging in clumps
        # =============================================================================

        for c, (new_clump, bound_ids) in enumerate(
            zip(start_of_new_clump, bound_star_ids), start=1
        ):
            if c == np.size(
                start_of_new_clump
            ):  # cheeky override once it reaches end of list
                star_ids_inside = bound_star_ids[new_clump:]
            else:
                star_ids_inside = bound_star_ids[new_clump : start_of_new_clump[c]]

            # get stats for a given pop2 clump, each value is a float
            # based solely on clump finder
            # haven't been profiled yet
            clump_idnum = clump_id[c - 1]
            clump_mask = np.isin(star_ids, star_ids_inside)

            clump_ctr_x = x[c - 1]
            clump_ctr_y = y[c - 1]
            clump_ctr_z = z[c - 1]
            rad = clump_rad[c - 1]
            # ages in Myr
            age = float(st.mode(pop2_ages[clump_mask], keepdims=True)[0])
            oldest_star = np.max(pop2_ages[clump_mask])
            youngest_star = np.min(pop2_ages[clump_mask])
            clump_mass = np.sum(pop2_masses[clump_mask])

            # clump_integrated_light = np.sum(10 ** pop2_lums[clump_mask])
            # 1d velovity dispersions
            std_vx = np.std(pop2_vx[clump_mask])
            std_vy = np.std(pop2_vy[clump_mask])
            std_vz = np.std(pop2_vz[clump_mask])

            # save each star position for profiler, recentered on the clump center
            clump_x = pop2_x[clump_mask] - clump_ctr_x
            clump_y = pop2_y[clump_mask] - clump_ctr_y
            clump_z = pop2_z[clump_mask] - clump_ctr_z

            # plt.figure()
            # plt.plot(log_bin_ctrs, surf_density)
            # plt.xscale("log")
            # plt.yscale("log")

            # the profiler can decide if it's a bsc or not
            disrupted_ids = []
            profiled_ids = []
            try:  # it can be fitted, otherwise no
                (
                    log_bin_ctrs,
                    surf_density,
                    surf_density_err,
                    r_half,
                    r_half_light,
                    m_clump,
                    lum_clump,
                ) = projected_surf_densities(
                    x_coord=clump_x,
                    y_coord=clump_y,
                    lums=10 ** pop2_lums[clump_mask],
                    masses=pop2_masses[clump_mask],
                    radius=rad,
                )

                fit_params, pcov = curve_fit(
                    f=modified_king_model,
                    xdata=log_bin_ctrs,
                    ydata=surf_density,
                    sigma=surf_density_err,
                    absolute_sigma=True,
                    p0=[1e4, 0.2, 2, 10],
                    bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, np.inf]),
                )
                central_dens, core_radius, alpha, bg_dens = fit_params
                cendens_err, corerad_err, alpha_err, bgdens_err = np.sqrt(np.diag(pcov))
                truncation_radius = trunc_radius(
                    central_dens, core_radius, alpha, bg_dens
                )
                if alpha > 8:
                    print("> Clump {} is being disrupted".format(clump_idnum))

                    disrupted_clump_data = pop2_data[:, 1:][clump_mask]
                    savename = os.path.join(
                        snapshot_container,
                        "disrupted_{}.txt".format(str(int(clump_idnum)).zfill(4)),
                    )
                    print("> Saving as {}".format(savename))
                    np.savetxt(
                        savename, X=disrupted_clump_data, header=pop2_header, fmt="%.6e"
                    )
                    disrupted_ids.append(star_ids_inside)

                else:
                    print("> Clump {} is a BSC".format(clump_idnum))
                    catalogue_to_save = [
                        clump_idnum,
                        clump_ctr_x,
                        clump_ctr_y,
                        clump_ctr_z,
                        rad,
                        age,
                        oldest_star,
                        youngest_star,
                        clump_mass,
                        # np.log10(clump_integrated_light),
                        std_vx,
                        std_vy,
                        std_vz,
                        core_radius,
                        corerad_err,
                        alpha,
                        alpha_err,
                        central_dens,
                        cendens_err,
                        bg_dens,
                        bgdens_err,
                        truncation_radius,
                    ]
                    reprocessed_clump_cata.append(catalogue_to_save)
                    bsc_profile_dat = np.vstack(
                        [log_bin_ctrs, surf_density, surf_density_err]
                    ).T

                    np.savetxt(
                        os.path.join(
                            snapshot_container,
                            "bsc_profile_{}.txt".format(str(int(clump_idnum)).zfill(4)),
                        ),
                        X=bsc_profile_dat,
                        header=r"radial_distnace [pc], sigma [msun/pc^2], err",
                        fmt="%.6e",
                    )

                    bsc_clump_dat = pop2_data[:, 1:][clump_mask]
                    np.savetxt(
                        os.path.join(
                            snapshot_container,
                            "bsc_{}.txt".format(str(int(clump_idnum)).zfill(4)),
                        ),
                        X=bsc_clump_dat,
                        header=pop2_header,
                        fmt="%.6e",
                    )
                    profiled_ids.append(star_ids_inside)
            except:
                pass
                print(" ")
                print("> Clump {} is not a bound star cluster".format(clump_idnum))
                print(" ")

            # segregate field stars and bound stars
        reprocessed_clump_cata = np.array(reprocessed_clump_cata)
        reprocessed_clump_header = (
            "clump_id \t x[pc] \t y[pc] \t z[pc] \t rad[pc] \t"
            "age [myr] \t oldest [myr] \t youngest [myr] \t mass [msun] \t "
            "std_vx [km/s] \t std_vy [km/s] \t std_vz [km/s] \t "
            "core_radius [pc] \t err \t "
            "alpha \t err \t "
            "central_dens [msun/pc^2] \t err \t "
            "bg_dens [msun/pc^2] \t err \t "
            "trunc rad [msun/pc^2]"
        )

        current_time = float(ds.current_time.in_units("Myr"))
        redshft = ds.current_redshift
        save_time = "{:.2f}".format(current_time).replace(".", "_")
        save_redshift = "{:.3f}".format(redshft).replace(".", "_")

        np.savetxt(
            os.path.join(
                snapshot_container,
                "profiled_catalogue-{}-{}-myr-z-{}.txt".format(
                    snap_strings[i], save_time, save_redshift
                ),
            ),
            X=reprocessed_clump_cata,
            header=reprocessed_clump_header,
            fmt="%.6e",
        )

        clumped_mask = np.isin(star_ids, bound_star_ids)
        all_clumped_stars = pop2_data[:, 1:][clumped_mask]
        all_field_stars = pop2_data[:, 1:][~clumped_mask]
        all_profiled_stars = pop2_data[:, 1:][np.isin(star_ids, profiled_ids)]
        all_disrupted_stars = pop2_data[:, 1:][np.isin(star_ids, disrupted_ids)]

        np.savetxt(
            os.path.join(
                snapshot_container,
                "clumped_stars-{}-{}-myr-z-{}.txt".format(
                    snap_strings[i], save_time, save_redshift
                ),
            ),
            X=all_clumped_stars,
            header=pop2_header,
            fmt="%.6e",
        )

        np.savetxt(
            os.path.join(
                snapshot_container,
                "field_stars-{}-{}-myr-z-{}.txt".format(
                    snap_strings[i], save_time, save_redshift
                ),
            ),
            X=all_field_stars,
            header=pop2_header,
            fmt="%.6e",
        )

        # clump finder mass, regardless if can be profiled
        tot_m_clumped = np.sum(all_clumped_stars[:, 9])
        # mass in BSC, which is clumps that can be profiled
        tot_m_bsc = np.sum(all_profiled_stars[:, 9])
        tot_m_disrupted = np.sum(all_disrupted_stars[:, 9])
        # mass in stars in the field component
        unbound_mass = np.sum(all_field_stars[:, 9])

        tot_lum_clumped = np.sum(10 ** all_clumped_stars[:, 2])
        tot_lum_bsc = np.sum(10 ** all_profiled_stars[:, 2])
        tot_lum_disrupted = np.sum(10 ** all_disrupted_stars[:, 2])
        unbound_lum = np.sum(10 ** all_field_stars[:, 2])
        # save simulation wide data;e.g. for bound/unbound over time
        tseries_header = (
            "snapshot time[myr] redshift clumped_mass[msun] "
            "bsc_mass[msun] disrupted_mass[msun] unbound_mass[msun] "
            "clumped_lum UV (150nm) [erg/s]"
            "total_lum UV (150nm) [erg/s] "
            "disrupted_lum UV (150nm) [erg/s] "
            "unbound_lum UV (150nm) [erg/s] "
        )
        save_per_snap = [
            float(snap_strings[i]),
            current_time,
            redshft,
            tot_m_clumped,
            tot_m_bsc,
            tot_m_disrupted,
            unbound_mass,
            tot_lum_clumped,
            tot_lum_bsc,
            tot_lum_disrupted,
            unbound_lum,
        ]

        time_series_data.append(save_per_snap)

        np.savetxt(
            os.path.join(
                bsc_cat_container,
                "{}_timeseries-{}-{}.txt".format(
                    sim_run, snap_strings[0], snap_strings[-1]
                ),
            ),
            X=np.array(time_series_data),
            header=tseries_header,
            fmt="%.6e",
        )

        print(f"> updated table with snapshot_{snap_strings[i]} data")
