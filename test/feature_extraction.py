import warnings

import bow as bow
import cv2
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.cluster import MiniBatchKMeans

logging.getLogger("snr")


def to_gray(color_img):
    """

    :param color_img:  image read by cv2
    :return: image converted to grayscale
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    """

    :param gray_img: grayscale image
    :return: keypoints, SIFT descriptors
    """
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    logging.debug("inside show sift features")
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


def match_2_sift_photos(photo1, photo1_desc, photo1_kp, photo2, photo2_desc, photo2_kp, top_N_match):
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


def show_sift_extracted(photo_path):
    photo = cv2.imread(photo_path)
    photo_gray = to_gray(photo)
    photo_kp, photo_desc = gen_sift_features(photo_gray)
    logging.info(
        "Photo: {photo_name} example descriptor: {desc}".format(photo_name=os.path.basename(photo1_path), desc=photo_desc[0]))
    show_sift_features(photo_gray, photo, photo_kp)
    # plt.show()
    return photo, photo_kp, photo_desc


# ToDO: srun cluster_and_split for all descriptors
# # generate indexes for train/test/val split
# training_idxs, test_idxs, val_idxs = search.bow.train_test_val_split_idxs(
#     total_rows=len(img_descs),
#     percent_test=0.15,
#     percent_val=0.15
# )
def cluster_and_split(img_descs, y, training_idxs, test_idxs, val_idxs, K):
    """Cluster into K clusters, then split into train/test/val"""
    # MiniBatchKMeans annoyingly throws tons of deprecation warnings that fill up the notebook. Ignore them.
    warnings.filterwarnings('ignore')

    X, cluster_model = bow.cluster_features(
        img_descs,
        training_idxs=training_idxs,
        cluster_model=MiniBatchKMeans(n_clusters=K)
    )

    warnings.filterwarnings('default')

    X_train, X_test, X_val, y_train, y_test, y_val = bow.perform_data_split(X, y, training_idxs, test_idxs, val_idxs)

    return X_train, X_test, X_val, y_train, y_test, y_val, cluster_model


if __name__ == "__main__":

    logging.basicConfig(filename="sift.log", level=logging.DEBUG)
    # important: write file path in double-quotes!
    # ToDO: change paths!
    photo1_path = "/home/monikas/Desktop/studia/SNR/SET_B/0368/00f163e3707840d1a1a5f2c93be77d16.jpg"
    # photo2_path = "/home/monikas/Desktop/studia/SNR/SET_B/0373/51056747963b4b27a10b7fcd2c79a135.jpg"
    photo2_path = "/home/monikas/Desktop/studia/SNR/SET_B/0368/1bb20a82e4b648d1908007e512c3455a.jpg"

    photo1, photo1_kp, photo1_desc = show_sift_extracted(photo1_path)
    photo2, photo2_kp, photo2_desc = show_sift_extracted(photo2_path)

    top_N_features_matched = 10
    match_2_sift_photos(photo1, photo1_desc, photo1_kp, photo2, photo2_desc, photo2_kp, top_N_features_matched)




