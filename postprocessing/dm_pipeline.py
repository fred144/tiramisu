import sys

sys.path.append("../")  # makes sure that importing the modules work
import numpy as np
import os

from src.lum.lum_lookup import lum_look_up_table
from tools.cosmo import code_age_to_myr
from tools.ram_fields import ram_fields
from tools.fscanner import filter_snapshots


import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(sys.argv[0], "usage:")
        print(
            "{} data_directory_to_postprocess start_snapshot end_snapshot step".format(
                sys.argv[0]
            )
        )
        exit()

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

    sim_run = datadir.replace("\\", "/").split("/")[-1]
    snaps, snap_strings = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )
    dm_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "dm_hop", sim_run
    )

    if not os.path.exists(dm_container):
        print("Creating container", dm_container)
        os.makedirs(dm_container)
    else:
        print("Tiramisu container found", dm_container)
        pass

    processor_number = 0
    cell_fields, epf = ram_fields()

    # %%
    m_vir = []
    r_vir = []
    tot_m_star = []
    t = []
    z = []
    snapshot = []

    for i, sn in enumerate(snaps):
        print(
            "# ________________________________________________________________________"
        )
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
        current_time = float(ds.current_time.in_units("Myr"))
        redshft = ds.current_redshift

        tot_m_star.append(np.sum(np.array(ad["star", "particle_mass"].to("Msun"))))

        t.append(current_time)
        z.append(redshft)
        m_vir.append(dm_halo_m)
        r_vir.append(vir_rad)
        snapshot.append(int(snap_strings[i]))
        dm_total_data = np.vstack((snapshot, t, z, m_vir, r_vir, tot_m_star)).T

        np.savetxt(
            os.path.join(dm_container, "main_halo.txt"),
            X=dm_total_data,
            header="snap_num\tt[myr]\tredshift\tmvir[msun]\trvir[pc]\tm_star[msun]",
        )
        print(">> table refreshed")
