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

    sim_run = datadir.replace("\\", "/").split("/")[-1]
    snaps, snap_strings = filter_snapshots(
        datadir,
        int(sys.argv[2]),
        int(sys.argv[3]),
        sampling=int(sys.argv[4]),
        str_snaps=True,
    )
    container = os.path.join(
        "..", "..", "container_tiramisu", "post_processed", "pop2", sim_run
    )

    if not os.path.exists(container):
        print("Creating container", container)
        os.makedirs(container)
    else:
        pass

    for i, sn in enumerate(snaps):
        print(
            "# ________________________________________________________________________"
        )
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings[i]}.txt"))

        print("# reading in", infofile)
