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


def get_total_img_cnt(dataset_path):
    counter = 0
    for folder_path in gen_subdir_path(dataset_path):
        for _, file_name in gen_file_path(folder_path):
            counter += 1
    return counter


def get_total_cls_cnt(dataset_path):
    counter = 0
    for _ in gen_subdir_path(dataset_path):
        counter += 1
    return counter


def scan_content(dataset_path):
    total_img_cnt = get_total_img_cnt(dataset_path)
    content = [('folder_path', 'img_name') for _ in range(total_img_cnt)]

    idx = 0
    for folder_path in gen_subdir_path(dataset_path):
        print(folder_path)
        for _, file_name in gen_file_path(folder_path):
            content[idx] = (folder_path, file_name)
            idx += 1
    return content


def remove_extension(file_name):
    return file_name.split(".")[0]


def add_folder(path, folder_name):
    return os.path.join(path, folder_name)


def get_folder(path):
    return os.path.basename(os.path.normpath(path))
