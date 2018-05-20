import cv2
import h5py
import logging
import matplotlib.pyplot as plt
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from sklearn.model_selection import train_test_split

from src.bounding_box import BoundingBox
from src import config


def match_2_sift_photos(photo1, photo1_kp, photo1_desc, photo2, photo2_kp, photo2_desc, top_n_match):
    """

    :param photo1: 1st photo
    :param photo1_kp: 1st photo keypoints
    :param photo1_desc: 1st photo descriptors
    :param photo2: 2nd photo
    :param photo2_kp: 2nd photo keypoints
    :param photo2_desc: 2nd photo descriptors
    :param top_n_match: number of top matched features that sould be visualized
    :return:
    """
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
        matches[:top_n_match], photo2.copy(), flags=0)

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
    return generate_image_descriptors(photo_cropped)


def generate_image_descriptors(photo_cropped):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    results = list()

    x = img_to_array(photo_cropped)
    x = x.reshape((1,) + x.shape)

    batch = datagen.flow(x, batch_size=config.data_multiplication_factor)
    for i in range(0, config.data_multiplication_factor + 1):
        pic = next(batch)
        pic = pic.reshape(pic.shape[1:4])
        pic = pic.astype(np.uint8, copy=False)
        _, photo_desc = gen_sift_features(pic)
        results.append(photo_desc)

    return results


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
        value = os.path.join(dir_path, sub_dir)
        yield value


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
        yield os.path.join(dir_path, file), file


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
            img_hash = tokens[0].replace("-", "")
            bounding_boxes[img_hash] = BoundingBox(tokens[1:])
    return bounding_boxes


def divide_data(features_db):
    labels_list = []
    ids = []
    ids_ref = dict()
    for class_name in features_db:
        for photo_name in features_db[class_name]:
            photo_name = photo_name.split('_')[0]
            if photo_name in ids_ref:
                continue
            ids.append((class_name, photo_name))
            ids_ref[photo_name] = 0
            labels_list.append(class_name)
    split_ids = {'training': {}, 'validation': {}, 'test': {}}

    split_ratio = config.training_total_ratio + config.validation_total_ratio
    split_ids['training']['data'], \
    split_ids['test']['data'], \
    split_ids['training']['labels'], \
    split_ids['test']['labels'] = train_test_split(ids, labels_list,
                                                   train_size=split_ratio,
                                                   stratify=labels_list,
                                                   random_state=0)

    split_ratio = config.training_total_ratio / (config.training_total_ratio + config.validation_total_ratio)
    split_ids['training']['data'], \
    split_ids['validation']['data'], \
    split_ids['training']['labels'], \
    split_ids['validation']['labels'] = train_test_split(split_ids['training']['data'],
                                                         split_ids['training']['labels'],
                                                         train_size=split_ratio,
                                                         stratify=split_ids['training']['labels'],
                                                         random_state=0)

    for group_name in split_ids.keys():
        print(group_name)
        group_db = h5py.File(config.groups_db_path[group_name], 'w')
        for photo in split_ids[group_name]['data']:
            class_name = photo[0]
            if class_name not in group_db.keys():
                group_db.create_group(class_name)
            for i in range(0, config.data_multiplication_factor):
                photo_name = photo[1] + "_" + str(i)
                group_db[class_name].create_dataset(photo_name, data=features_db[class_name][photo_name])
        group_db.close()


if __name__ == "__main__":
    logging.basicConfig(filename="sift.log", level=logging.DEBUG)

    features_db = h5py.File(config.features_db_path, "w")
    bounding_boxes = get_bounding_boxes(config.bounding_boxes_path)

    counter = 0
    logging.info("Starting extraction")
    for class_path in generate_subdir_path(config.set_path):
        class_descriptors = features_db.create_group(os.path.basename(os.path.normpath(class_path)))
        for photo_path, photo_name in generate_file_path(class_path):
            counter += 1
            if counter % config.take_every_nth_sample != 0:
                continue
            # removes file extension
            photo_name_hash = photo_name.split(".")[0]
            bb = bounding_boxes[photo_name_hash]
            photo_desc = execute_sift_extraction(photo_path, bb, 1)
            for i, pic in enumerate(photo_desc):
                class_descriptors.create_dataset(photo_name_hash + "_" + str(i), data=photo_desc[i])
    features_db.close()
    logging.info("Extraction finished")

    logging.info("Dividing data into groups")
    features_db = h5py.File(config.features_db_path, "r")
    divide_data(features_db)
    features_db.close()
    logging.info("Dividing finished")
