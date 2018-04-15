import cv2
import logging
import matplotlib.pyplot as plt
import os

from .bounding_box import BoundingBox
from klepto.archives import dir_archive
from src import config


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


def execute_sift_extraction(photo_path, bounding_box, with_colour=1):
    """

    :param photo_path: self explanatory
    :param bounding_box: bounding box object
    :param with_colour: set 1 if request RGB image or set 0 if request gray_scale image
    :return: photo, keypoints, SIFT_descriptors
    """
    logging.info("Starting extraction for photo: {photo_name}".format(photo_name=photo_path))
    photo = cv2.imread(photo_path, with_colour)
    photo_cropped = photo[bounding_box.y0:bounding_box.y0 + bounding_box.dy,
                    bounding_box.x0:bounding_box.x0 + bounding_box.dx]
    photo_kp, photo_desc = gen_sift_features(photo_cropped)
    return photo_cropped, photo_kp, photo_desc


def generate_subdir_path(dir_path):
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
        yield dir_path + sub_dir + '/'


def generate_file_path(dir_path):
    """
    generates paths to files in selected directory
    :param dir_path: directory path
    :return: path to file and file name
    """
    files = next(os.walk(dir_path))[2]
    for file in files:
        if file == ".directory":
            continue
        yield (dir_path + file), file


def get_bounding_boxes(file_path):
    """

    :param file_path: path to file containing data
    :return: dictionary of bounding boxes with key as image name
    """
    logging.info("Loading bounding boxes")
    bounding_boxes = dict()
    with open(file_path) as file:
        for raw_line in file.readlines():
            tokens = raw_line.strip().split(' ')
            hash = tokens[0].replace("-", "")
            bounding_boxes[hash] = BoundingBox(tokens[1:])
    return bounding_boxes


if __name__ == "__main__":
    logging.basicConfig(filename="sift.log", level=logging.DEBUG)

    bounding_boxes = get_bounding_boxes(config.bounding_boxes_path)

    features_db = dir_archive(config.features_db_path, {}, serialized=True, cached=False)
    for class_name in next(os.walk(config.set_path))[1]:
        features_db[class_name] = {}

    counter = 0
    logging.info("Starting extraction")
    for class_path in generate_subdir_path(config.set_path):
        print(class_path)
        class_descriptors = {}
        for photo_path, photo_name in generate_file_path(class_path):
            counter += 1
            if counter % config.take_every_nth_sample != 0:
                continue
            # removes file extension
            photo_name_hash = photo_name.split(".")[0]
            bb = bounding_boxes[photo_name_hash]
            photo1, photo1_kp, photo1_desc = execute_sift_extraction(photo_path, bb, 1)
            class_descriptors[photo_name_hash] = photo1_desc
        features_db[os.path.basename(os.path.normpath(class_path))] = class_descriptors
    logging.info("Extraction finished")
