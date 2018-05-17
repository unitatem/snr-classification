import logging

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as image_manip

import config
import file
from bounding_box import BoundingBox


def plot_img(img):
    plt.imshow(img)
    plt.show()


def preprocess_image(image):
    x = image_manip.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def main():
    logging.basicConfig(filename=__file__ + '.log', level=logging.DEBUG)

    bounding_boxes = BoundingBox.get_bounding_boxes(config.bounding_boxes_path)
    preprocessed_dataset_db = h5py.File(config.resources_path + "preprocessed.hdf5", "w")

    logging.info("Starting image loading")
    counter = 0
    for class_path in file.gen_subdir_path(config.set_path):
        print(class_path)
        class_descriptors = preprocessed_dataset_db.create_group(file.get_dst_folder(class_path))
        for photo_path, photo_name in file.gen_file_path(class_path):
            counter += 1
            if counter % config.take_every_nth_sample != 0:
                continue

            img = cv2.imread(photo_path, 1)

            photo_name_hash = file.remove_extension(photo_name)
            bb = bounding_boxes[photo_name_hash]

            img_cropped = img[bb.y0:bb.y0 + bb.dy, bb.x0:bb.x0 + bb.dx]
            img_resized = cv2.resize(img_cropped, (224, 224))
            img_preprocessed = preprocess_image(img_resized)

            class_descriptors.create_dataset(photo_name_hash, data=img_preprocessed)

    preprocessed_dataset_db.close()
    logging.info("Image loading finished")


if __name__ == "__main__":
    main()
