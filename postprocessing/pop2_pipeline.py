import sys

sys.path.append("..")  # makes sure that importing the modules work
import numpy as np
import os

from src.lum.lum_lookup import lum_look_up_table
from tools.cosmo import code_age_to_myr
from tools.ram_fields import ram_fields
from tools.fscanner import filter_snapshots


import yt


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(sys.argv[0], "usage:")
        print(
            "{} data_directory_to_postprocess start_snapshot end_snapshot step".format(
                sys.argv[0]
            )
        )
        exit()

    cell_fields, epf = ram_fields()
    # datadir = os.path.relpath("../../cosm_test_data/refine")
    datadir = sys.argv[1]

    # datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")

    sim_run = os.path.basename(os.path.normpath(datadir))

    print("- sim run", sim_run)
    snaps, snap_strings = filter_snapshots(
        datadir,
        int(sys.argv[2]),
        int(sys.argv[3]),
        sampling=int(sys.argv[4]),
        str_snaps=True,
    )
    pop2_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "pop2", sim_run
    )

    if not os.path.exists(pop2_container):
        print("Creating container", pop2_container)
        os.makedirs(pop2_container)
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
        print(
            "# ________________________________________________________________________"
        )
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))

        print("# reading in", infofile)

        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        # get time-dependent params.
        redshft = ds.current_redshift
        current_hubble = ds.hubble_constant
        current_time = float(ds.current_time.in_units("Myr"))

        # get SFC/PSC positions and other important fields,
        # need to modify definitions to get these sinks
        # pos_sfcs = np.array(ad["SFC", "particle_position"])
        # pos_pscs = np.array(ad["PSC", "particle_position"])
        # pos_sfcs_recentered = pos_sfcs - plt_ctr
        # pos_pscs_recentered = pos_pscs - plt_ctr

        # read POPII star info
        star_id = np.array(ad["star", "particle_identity"])
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])

        if len(star_id) == 0:
            print("- no particle data to extract")
            continue

        # center based on star position distribution
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)

        plt_ctr = np.array([x_center, y_center, z_center])
        plt_ctr_in_pc = np.array(ds.arr(plt_ctr, "code_length").to("pc"))

        # translate points to stellar CoM
        x_pos = x_pos - plt_ctr[0]
        y_pos = y_pos - plt_ctr[1]
        z_pos = z_pos - plt_ctr[2]

        # x_vel = ad["star", "particle_velocity_x"].to("km/s")
        # y_vel = ad["star", "particle_velocity_y"].to("km/s")
        # z_vel = ad["star", "particle_velocity_z"].to("km/s")

        #  converts code age to relative ages
        # calculate the age of the universe when the first star was born
        # using the logSFC as a reference point for redshift when the first star
        # was born. Every age is relative to this. Due to our mods of ramses.

        first_form = np.loadtxt(sim_logfile, usecols=2).max()
        birth_start = np.round_(
            float(ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
        )

        # all the birth epochs of the stars
        converted_unfiltered = code_age_to_myr(
            ad["star", "particle_birth_epoch"],
            current_hubble,
            unique_age=False,
        )

        # ==========================luminosity mappping data extraction==============

        birthtime = np.round(converted_unfiltered + birth_start, 3)  #!
        current_ages = np.array(np.round(current_time, 3) - np.round(birthtime, 3))
        # import time

        # s = time.perf_counter()
        pop2_lums = (
            lum_look_up_table(
                stellar_ages=current_ages * 1e6,
                table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
                column_idx=1,
                log=True,
            )
            - 5  # since we are using 10^6 M_sun for the starburst
        )
        # e = time.perf_counter()
        # print(e - s)
        snap_info = np.array(
            [
                np.concatenate(
                    (np.array([current_time, redshft]), plt_ctr, plt_ctr_in_pc)
                )
            ]
        )
        snap_info.resize(np.size(current_ages))
        star_info = np.array(
            [
                snap_info,
                star_id,
                current_ages,
                pop2_lums,
                ds.arr(x_pos, "code_length").to("pc"),
                ds.arr(y_pos, "code_length").to("pc"),
                ds.arr(z_pos, "code_length").to("pc"),
                ad["star", "particle_velocity_x"].to("km/s"),
                ad["star", "particle_velocity_y"].to("km/s"),
                ad["star", "particle_velocity_z"].to("km/s"),
                ds.arr(ad["star", "particle_mass"], "code_mass").to("msun"),
            ]
        ).T

        # =========================star positions save=================================

        save_time = "{:.2f}".format(current_time).replace(".", "_")
        save_redshift = "{:.3f}".format(redshft).replace(".", "_")
        save_name = os.path.join(
            pop2_container,
            "pop2-{}-{}-myr-z-{}.txt".format(snap_strings[i], save_time, save_redshift),
        )

        header = (
            "|t_sim[Myr],z,ctr(code),ctr(pc)|"
            "|ID"
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
        np.savetxt(save_name, X=star_info, header=header, fmt="%.6e")

        print("# saved:", save_name)
