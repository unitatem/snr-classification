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


# ToDO: might be improved!
def get_total_img_cnt(dataset_path):
    counter = 0
    for folder_path in gen_subdir_path(dataset_path):
        _, _, files = next(os.walk(folder_path))
        counter += len(files)
    return counter


def get_total_cls_cnt(dataset_path):
    """

    :param dataset_path: dataset path
    :return: total number of classes in the whole dataset
    """
    _, dirs, _ = next(os.walk(dataset_path))
    dir_count = len(dirs)
    print(dir_count)
    return dir_count


def scan_content(dataset_path):
    content = []
    for folder_path in gen_subdir_path(dataset_path):
        for _, file_name in gen_file_path(folder_path):
            content.append((folder_path, file_name))
    return content


def remove_extension(file_name):
    return file_name.split(".")[0]


def add_folder(path, folder_name):
    return os.path.join(path, folder_name)


def get_folder(path):
    return os.path.basename(os.path.normpath(path))
