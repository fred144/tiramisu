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


def find_matching_time(sequence, look_up_sequence, orig_seq_out_num=False, fmt="Myr"):
    r"""
    given a sequence, return files from look_up sequence with the same time
    Example:

        fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 113, 1000, 1)
        fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 917, 1)
        f3_matched = find_matching_time(fs035, fs070)
    """
    look_up_sequence_output_nums = []
    look_up_sequence_output_times = []
    for file_name in look_up_sequence:
        # print(file_name)
        # try:
        out_num = int(file_name.split("/")[-1].split("-")[1])
        # except:
        #     out_num = int(file_name.split("\\")[-1].split("_")[1])
        sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
        look_up_sequence_output_nums.append(out_num)
        look_up_sequence_output_times.append(sim_time)
    look_up_sequence_output_nums = np.array(look_up_sequence_output_nums)
    look_up_sequence_output_times = np.array(look_up_sequence_output_times)

    if fmt != "Myr":
        sequence_output_nums = []
        sequence_output_times = []
        for file_name in sequence:
            sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
            # sim_time = float(file_name.split("/")[-1][10:16].replace("_", "."))
            sequence_output_nums.append(out_num)
            sequence_output_times.append(sim_time)
        sequence_output_nums = np.array(sequence_output_nums)
        sequence_output_times = np.array(sequence_output_times)
    else:
        sequence_output_times = np.array(sequence)

    # since fs70 has progressed more, use it as a lookup table to match fs035
    residuals = np.abs(
        look_up_sequence_output_times - sequence_output_times[:, np.newaxis]
    )
    closest_match_idxs = np.argmin(residuals, axis=1)

    look_up_sequence_same_time = look_up_sequence_output_nums[closest_match_idxs]

    filter_list = list(map(str, list(look_up_sequence_same_time)))
    filter_list = [f.zfill(5) for f in filter_list]

    common_items = []
    for num in filter_list:
        for file in look_up_sequence:
            if num in file:
                common_items.append(file)
                break

    if orig_seq_out_num is True and fmt != "Myr":
        orig_sequence_output_nums = []
        orig_sequence_output_times = []
        for file_name in sequence:
            try:
                out_num = int(file_name.split("/")[-1].split("_")[1])
            except:
                out_num = int(file_name.split("\\")[-1].split("_")[1])
            sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
            orig_sequence_output_nums.append(out_num)
            orig_sequence_output_times.append(sim_time)
        orig_sequence_output_nums = np.array(orig_sequence_output_nums)
        orig_sequence_output_times = np.array(orig_sequence_output_times)

        return common_items, look_up_sequence_same_time, orig_sequence_output_nums
    else:
        return common_items, look_up_sequence_same_time


# fpathstest, _ = filter_snapshots(
#     "/home/fabg/container_tiramisu/post_processed/pop2/CC-Fiducial/",
#     304,
#     466,
#     sampling=1,
#     str_snaps=True,
#     snapshot_type="pop2_processed",
# )
# find_matching_time([565], fpathstest, orig_seq_out_num=False, fmt="Myr")
