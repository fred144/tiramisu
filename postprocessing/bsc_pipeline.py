import sys

sys.path.append("..")  # makes sure that importing the modules work
import numpy as np
import os
import h5py as h5
from src.lum.lum_lookup import lum_look_up_table
from tools.cosmo import code_age_to_myr
from tools.ram_fields import ram_fields
from tools.fscanner import filter_snapshots


import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print(sys.argv[0], "usage:")
    #     print(
    #         "{} data_directory_to_postprocess start_snapshot end_snapshot step".format(
    #             sys.argv[0]
    #         )
    #     )
    #     exit()

    # # datadir = os.path.relpath("../../cosm_test_data/refine")
    # datadir = sys.argv[1]

    # local path for test
    # datadir = os.path.relpath("../../sim_data/cluster_evolution/CC-radius1")
    datadir = os.path.relpath("../../garcia23_testdata/fs07_refine")
    start_snapshot = 500
    end_snapshot = 500
    step = 1
    sim_run = datadir.replace("\\", "/").split("/")[-1]
    snaps, snap_strings = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
    )

    bsc_container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "bsc", sim_run
    )

    if not os.path.exists(bsc_container):
        print("Creating container", bsc_container)
        os.makedirs(bsc_container)
    else:
        pass

    processor_number = 0
    cell_fields, epf = ram_fields()

    tot_m_bsc = []

    tot_m_star = []
    t = []
    z = []
    snap = []

    for i, sn in enumerate(snaps):
        print(
            "# ________________________________________________________________________"
        )
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))

        print("# reading in", infofile)

        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        fof_catalogue = "{}/info_{}/info_{}.{}.h5".format(
            bsc_container,
            snap_strings[i],
            snap_strings[i],
            processor_number,
        )

        if os.path.isfile(fof_catalogue) is True:
            print("file already exists")
            pass
        else:
            hc = HaloCatalog(
                data_ds=ds,
                finder_method="fof",
                finder_kwargs={
                    "ptype": "star",
                    "padding": 0.0001,
                    "link": 0.00001,  # "best"
                    # "link": 0.0000025, # "fof"
                    "dm_only": False,
                },
            )
            hc.create()

        # read POPII star info
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])
        star_id = np.array(ad["star", "particle_identity"])

        # center based on star position distribution
        x_center = np.mean(x_pos).to("pc")
        y_center = np.mean(y_pos).to("pc")
        z_center = np.mean(z_pos).to("pc")
        plt_ctr = np.array([x_center, y_center, z_center])

        cata_h5 = h5.File(fof_catalogue, "r")

        # need to read in using yt for virial radius for some reason unknown units in catalogue
        cata_yt = yt.load(fof_catalogue)

        # make a halo catalogue for yt overplot
        halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
        halo_cat_plotting.load()

        cata_yt = cata_yt.all_data()

        # pop2_data = np.loadtxt(
        #     glob.glob(
        #         "../particle_data/pop_2_data/{}/pos_{}_*.txt".format(
        #             simulation_run, snapshot_num_string
        #         )
        #     )[0]
        # )
        # ctr_at_pc = pop2_data[5:8, 6]
        # ctr_at_code_length = pop2_data[2:5, 6]
        # star_ids = pop2_data[:, 0]
        # x_pos = pop2_data[:, 2]
        # y_pos = pop2_data[:, 3]
        # z_pos = pop2_data[:, 4]

        # get the halo centers

        halo_id = np.array(cata_h5["particle_identifier"])

        halo_x = np.array(cata_h5["particle_position_x"])
        halo_y = np.array(cata_h5["particle_position_y"])
        halo_z = np.array(cata_h5["particle_position_z"])

        halo_x = np.array(ds.arr(halo_x, "code_length").to("pc")) - ctr_at_pc[0]
        halo_y = np.array(ds.arr(halo_y, "code_length").to("pc")) - ctr_at_pc[1]
        halo_z = np.array(ds.arr(halo_z, "code_length").to("pc")) - ctr_at_pc[2]

        # get halo virial radii
        halo_vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"], "cm").to("pc"))

        cat_pc = np.vstack((halo_id, halo_x, halo_y, halo_z, halo_vir_rad)).T
        cat_save_name = "../halo_data/{}/{}/info_{}/catalogue_{}.txt".format(
            simulation_run,
            finder_profiler_run,
            snapshot_num_string,
            snapshot_num_string,
        )
        header = (
            "halo_id \t x_coord [pc] \t y_coord [pc] \t z_coord [pc] \t vir_rad [pc]"
        )
        np.savetxt(cat_save_name, X=cat_pc, header=header)

        # get particles belonging to each halo
        num_stars_in_halo = np.array(cata_h5["particle_number"])
        start_of_new_halo = np.array(cata_h5["particle_index_start"])
        halo_star_ids = np.array(cata_h5["particles/ids"])

        cata_h5.close()
        for i, (new_h, h_id) in enumerate(zip(start_of_new_halo, halo_id), start=1):
            if i == np.size(
                start_of_new_halo
            ):  # cheeky over ride once it reaches end of list
                star_ids_inside = halo_star_ids[new_h:]
            else:
                star_ids_inside = halo_star_ids[new_h : start_of_new_halo[i]]

            # translate stars, taking into account centers
            gc_mask = np.isin(star_ids, star_ids_inside)
            gc_x = x_pos[gc_mask]
            gc_y = y_pos[gc_mask]
            gc_z = z_pos[gc_mask]

            # take the x,y,z of individual clusters and subtract center of halo cluster
            # this makes them all centered at the origin (0,0,0)
            gc_stars = np.vstack((gc_x, gc_y, gc_z)).T - cat_pc[:, 1:-1][i - 1]
            gc_stars = np.column_stack((star_ids_inside, gc_stars))
            header = "star id \t star_x_coords [pc] \t star_y_coords [pc] \t star_z_coords [pc] "

            save_name = "../halo_data/{}/{}/info_{}/gc_vir_{}.txt".format(
                simulation_run,
                finder_profiler_run,
                snapshot_num_string,
                str(int(h_id)).zfill(3),
            )
            np.savetxt(save_name, X=gc_stars, header=header)
        # cata_yt = cata_yt.all_data()
        # dm_halo_m = np.max(np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun")))
        # haloidx = np.argmax(
        #     np.array(ds.arr(cata_yt["all", "particle_mass"]).to("Msun"))
        # )
        # vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"]).to("pc"))[haloidx]
        # m_vir.append(dm_halo_m)
        # r_vir.append(vir_rad)
