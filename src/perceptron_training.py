import logging
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from src import config
from random import shuffle


def build_perceptron():
    logging.info('Building perceptron: [' + ' '.join([str(x) for x in config.sizes_of_layers]) + ']')
    model = Sequential()
    for i, layer_size in enumerate(config.sizes_of_layers):
        if i == 0:
            model.add(Dense(layer_size, input_dim=config.clusters_count))
        else:
            model.add(Dense(layer_size))
        model.add(Activation('sigmoid'))
    cluster_db = h5py.File(config.clusters_db_path, 'r')
    output_size = len(cluster_db)
    cluster_db.close()
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_class_mapping():
    cluster_db = h5py.File(config.clusters_db_path, 'r')
    class_names = list(cluster_db.keys())
    sorted(class_names)
    mapping = {}
    for i, class_name in enumerate(class_names):
        mapping[class_name] = i
    cluster_db.close()
    return mapping


def divide_data():
    cluster_db = h5py.File(config.clusters_db_path)
    sample_ids = []
    for class_name in cluster_db:
        for photo_name in cluster_db[class_name]:
            sample_ids.append((class_name, photo_name))
    cluster_db.close()
    shuffle(sample_ids)
    divide_points = (len(sample_ids) * config.training_fraction,
                     len(sample_ids) * config.training_fraction + len(sample_ids) * config.test_fraction)
    training_ids = sample_ids[:divide_points[0]]
    test_ids = sample_ids[divide_points[0]:divide_points[1]]
    validation_ids = sample_ids[divide_points[1]:]

    return training_ids, validation_ids, test_ids


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)
    model = build_perceptron()
