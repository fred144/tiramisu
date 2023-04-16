"""
logfile scraper for lustre data 
"""
import numpy as np
import os
import sys
import shutil
import yt
from yt.funcs import mylog
import warnings


mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def snap_shot_info(sim_output_abs_dir, save_path):
    output_name = os.path.basename(os.path.normpath(sim_output_abs_dir))
    info_file_name = "info_" + output_name[7:] + ".txt"

    infofile = os.path.abspath(os.path.join(sim_output_abs_dir, info_file_name))
    print("# Reading in", infofile)

    # read fields explicitly, not recognized by YT from this ver of RAMSES
    cell_fields = [
        "Density",
        "x-velocity",
        "y-velocity",
        "z-velocity",
        "Pressure",
        "Metallicity",
        "dark_matter_density",
        "xHI",
        "xHII",
        "xHeII",
        "xHeIII",
    ]
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]

    # read in RAMSES data set
    ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
    redshft = ds.current_redshift
    current_time = float(ds.current_time.in_units("Myr"))

    ad = ds.all_data()

    # get indivudual masses
    dark_matter = ad["DM", "particle_mass"].in_units("Msun")
    pop_ii = ad["star", "particle_mass"].in_units("Msun")
    # living  pop three stars
    pop_iii = ad["POPIII", "particle_mass"].in_units("Msun")
    # Pop III stars taking place SNe
    sn = ad["supernova", "particle_mass"].in_units("Msun")
    # Pop III stars after SNe
    dead = ad["dead", "particle_mass"].in_units("Msun")
    # Pop III remnant BHs
    black_hole = ad["BH", "particle_mass"].in_units("Msun")
    # gas
    m_gas = ad["gas", "mass"].in_units("Msun").sum()

    # total masses
    m_dark_matter = dark_matter.sum()
    m_pop_ii = pop_ii.sum()
    m_pop_iii = pop_iii.sum()
    m_sn = sn.sum()
    m_dead = dead.sum()
    m_black_hole = black_hole.sum()

    # [redshift, current_time, dm, pop2, pop3, sn, dead, BH, gas mass]
    tot_masses = np.array(
        [
            redshft,
            current_time,
            m_dark_matter,
            m_pop_ii,
            m_pop_iii,
            m_sn,
            m_dead,
            m_black_hole,
            m_gas,
        ]
    )
    header = (
        "latest_output:{} \t z \t\t SimTime[Myr] \t DM[Msun] \t pop2[Msun] \t pop3[Msun]"
        " \t SN[Msun] \t Dead[Msun] \t BH[Msun] \t Gas [Msun]"
    ).format(output_name[7:])

    save_path = os.path.join(save_path, "latest_sim_stats.txt")
    np.savetxt(fname=save_path, X=np.atleast_2d(tot_masses), header=header)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(sys.argv[0], "usage:")
        print("{} data_directory_to_scrape".format(sys.argv[0]))
        exit()
    container = "../container_ram-py/"
    newpath = os.path.join(container)
    if not os.path.exists(newpath):
        print("====================================================")
        print("Creating ram-py container", newpath)
        print("====================================================")
        os.makedirs(newpath)

    # lustre = "/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/"
    run_directories = sys.argv[1]
    print(sys.argv[1])
    # get latest sim results
    cleaned_run_directories = next(os.walk(run_directories))[1]
    sim_runs = [x for x in cleaned_run_directories if "old" or "." not in x]
    print("> found the following runs")
    print(sim_runs)
    for path, subdirs, files in os.walk(run_directories):
        # path is lustre simulation run names, and output directories
        for file_name in files:
            # if it is a log file and not from old runs
            if "log" in file_name and "old" not in path:
                # full path to log files
                full_path_to_files = os.path.join(path, file_name)

                # get simulation run names; e.g., fs07_refine, fs035_ms10
                # last entry of directory abs path is the directory/ simulation run itself
                sim_name = os.path.basename(os.path.dirname(full_path_to_files))
                save_folder_name = os.path.join(container, sim_name)

                # folder for saving in the script directory
                if not os.path.exists(save_folder_name):
                    print("# Creating new directory:", save_folder_name)
                    os.makedirs(save_folder_name)

                # copy over the log file
                shutil.copy(src=full_path_to_files, dst=save_folder_name)

                print("# Copied: ", full_path_to_files)

    for simrun in sim_runs:
        sim_folder = os.path.join(run_directories, simrun)
        output_folders = sorted(os.listdir(sim_folder))
        output_folders = [x for x in output_folders if "output_" in x]
        latest_snapshots_abs_dir = os.path.join(sim_folder, output_folders[-1])
        # get the run name
        run_name = os.path.split(os.path.split(latest_snapshots_abs_dir)[0])[1]
        # save path based on the simulation run name
        save_path = os.path.join(container, run_name)
        print("# Saving latest results for", run_name)
        snap_shot_info(latest_snapshots_abs_dir, save_path)
