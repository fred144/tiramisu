import os
import numpy as np


def filter_snapshots(
    folder_path,
    start_snap: int,
    end_snap: int,
    sampling=1,
    str_snaps=False,
    snapshot_type="ramses_snapshot",
):
    r"""Given a directory of outputs, return a list of relative file
    paths given a range of snapshot values.

    Enables discrete selection of time range based on snapshot number.

    """

    files = sorted(os.listdir(folder_path))

    if snapshot_type == "ramses_snapshot":
        files = [x for x in files if "output_" in x]
        sn_nums = np.array([int(i.split("_")[-1]) for i in files])
    elif snapshot_type == "pop2_processed":
        sn_nums = np.array([int(i.split("-")[1]) for i in files])
    elif snapshot_type == "bsc_processed":
        files = [x for x in files if "info_" in x]
        # print(files)
        sn_nums = np.array([int(i.split("_")[-1]) for i in files])

    print("> running filter_snapshots")
    print("> processing {}".format(snapshot_type))
    print("> found", len(files), "snapshots")
    if np.isin(start_snap, sn_nums) and np.isin(end_snap, sn_nums):
        strt_string = str(start_snap).zfill(5)
        end_string = str(end_snap).zfill(5)

        strt_idx = [i for i, s in enumerate(files) if strt_string in s][0]
        end_idx = [i for i, s in enumerate(files) if end_string in s][0]
    else:
        print("* start or stop snapshot doesn't exist")
        print("> starting at", sn_nums.min())
        print("> stopping at", sn_nums.max())
        strt_string = str(sn_nums.min()).zfill(5)
        end_string = str(sn_nums.max()).zfill(5)

        strt_idx = [i for i, s in enumerate(files) if strt_string in s][0]
        end_idx = [i for i, s in enumerate(files) if end_string in s][0]

    filtered_files = files[strt_idx : end_idx + 1 : sampling]
    string_nums = [i.split("_")[-1] for i in files][strt_idx : end_idx + 1 : sampling]
    # print(string_nums)
    rel_paths = [os.path.join(folder_path, file) for file in filtered_files]

    print("> returning", len(filtered_files))
    print(">>> done ")
    if str_snaps == False:
        return rel_paths
    else:
        return rel_paths, string_nums
