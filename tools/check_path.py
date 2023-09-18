import os


def check_path(path):
    if not os.path.exists(path):
        print("> creating container", path)
        os.makedirs(path)
    else:
        pass
