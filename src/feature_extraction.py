import cv2
from klepto.archives import dir_archive
import logging
import matplotlib.pyplot as plt
import os

logging.getLogger("feature_extraction")


def match_2_sift_photos(photo1, photo1_kp, photo1_desc, photo2, photo2_kp, photo2_desc, top_N_match):
    # create a BFMatcher object which will match up the SIFT features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(photo1_desc, photo2_desc)

    # Sort the matches in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    logging.debug("Matches: {matches}".format(matches=len(matches)))

    # draw the top N matches
    match_img = cv2.drawMatches(
        photo1, photo1_kp,
        photo2, photo2_kp,
        matches[:top_N_match], photo2.copy(), flags=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(match_img)
    plt.show()


def gen_sift_features(img):
    """

    :param img: image
    :return: keypoints, SIFT descriptors
    """
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def execute_sift_extraction(photo_path, with_colour=1):
    """

    :param photo_path: self explanatory
    :param with_colour: set 1 if request RGB image or set 0 if request gray_scale image
    :return: photo, keypoints, SIFT_descriptors
    """
    logging.info("Starting extraction for photo: {photo_name}".format(photo_name=photo_path))
    photo = cv2.imread(photo_path, with_colour)
    photo_kp, photo_desc = gen_sift_features(photo)
    return photo, photo_kp, photo_desc


def generate_subdir_path(dir_path):
    """
    generates path to subdirectories in selected directory
    :param dir_path: directory path
    :return: path to subdir
    """
    # os.walk is a generator -> next gives first tuple -> second element is list of all subdir
    subdirs = next(os.walk(dir_path))[1]
    for subdir in subdirs:
        if subdir == ".directory":
            continue
        yield dir_path + subdir + '/'


def generate_file_path(dir_path):
    """
    generates paths to photos in selected directory
    :param set_path: important: write file path in double-quotes
    :return: path to photo and class name
    """
    files = next(os.walk(dir_path))[2]
    for file in files:
        if file == ".directory":
            continue
        yield (dir_path + file), os.path.basename(os.path.normpath(dir_path))


if __name__ == "__main__":
    logging.basicConfig(filename="sift.log", level=logging.DEBUG)
    resources_path = "../resources/"
    set_path = resources_path + "SET_B/"

    features_db = dir_archive(resources_path + "extracted_features.db", {}, serialized=True, cached=False)

    for class_name in next(os.walk(set_path))[1]:
        features_db[class_name] = []

    logging.info("Starting extraction")
    for class_path in generate_subdir_path(set_path):
        print(class_path)
        class_descriptors = []
        for photo_path, class_name in generate_file_path(class_path):
            photo1, photo1_kp, photo1_desc = execute_sift_extraction(photo_path, 1)
            class_descriptors.append(photo1_desc)
        features_db[os.path.basename(os.path.normpath(class_path))] = class_descriptors
    logging.info("Extraction finished")

    # del db
    # db = dir_archive('extracted_features.db', {}, serialized=True)
    # db.load()

