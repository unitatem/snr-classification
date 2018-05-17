import os


def gen_subdir_path(dir_path):
    """
    generates path to subdirectories in selected directory
    :param dir_path: directory path
    :return: path to subdir
    """
    # os.walk is a generator -> next gives first tuple -> second element is list of all subdir
    sub_dirs = next(os.walk(dir_path))[1]
    for sub_dir in sub_dirs:
        if sub_dir == ".directory":
            continue
        yield os.path.join(dir_path, sub_dir)


def gen_file_path(dir_path):
    """
    generates paths to files in selected directory
    :param dir_path: directory path
    :return: path to file and file name
    """
    files = next(os.walk(dir_path))[2]
    for file in files:
        if file == ".directory":
            continue
        yield os.path.join(dir_path, file), file


def remove_extension(file_name):
    return file_name.split(".")[0]


def get_dst_folder(path):
    return os.path.basename(os.path.normpath(path))
