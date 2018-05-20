import logging
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as image_manip

import config
import divide_dataset
import file
from bounding_box import BoundingBox


def plot_img(img):
    plt.imshow(img)
    plt.show()


def preprocess_image(image):
    x = cv2.resize(image, (224, 224))
    x = image_manip.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_and_preprocess_img(folder_path, img_name, bounding_boxes):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)

    img_name_hash = file.remove_extension(img_name)
    bb = bounding_boxes[img_name_hash]
    img_cropped = img[bb.y0:bb.y0 + bb.dy, bb.x0:bb.x0 + bb.dx]

    img_preprocessed = preprocess_image(img_cropped)
    return img_preprocessed


def main():
    logging.basicConfig(filename=__file__ + '.log', level=logging.DEBUG)

    logging.info("Scanning content of dataset")
    content = file.scan_content(config.set_path)

    logging.info("Dividing data into groups")
    divided_content = divide_dataset.divide(content)

    bounding_boxes = BoundingBox.get_bounding_boxes(config.bounding_boxes_path)
    logging.info("Starting image preprocessing")
    counter = 0
    for key in divided_content.keys():
        database = h5py.File(config.get_convolution_datasets_path(key), 'w')
        for (cls_name, img_name) in divided_content[key]:
            counter += 1
            if counter % config.take_every_nth_sample != 0:
                continue

            if cls_name not in database.keys():
                database.create_group(cls_name)
            cls_path = file.add_folder(config.set_path, cls_name)
            database[cls_name].create_dataset(file.remove_extension(img_name),
                                              data=load_and_preprocess_img(cls_path, img_name, bounding_boxes))
        database.close()
    logging.info("Image loading finished")


if __name__ == "__main__":
    main()
