import logging
import h5py
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
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    cluster_db.close()
    return model


def divide_data(training_fraction):
    cluster_db = h5py.File(config.clusters_db_path, 'r')
    data_ids = []
    for class_name in cluster_db:
        for photo_name in cluster_db[class_name]:
            data_ids.append((class_name, photo_name))
    cluster_db.close()

    shuffle(data_ids)
    divide_point = int(len(data_ids)*training_fraction)
    return data_ids[:divide_point], data_ids[divide_point:]


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)
    model = build_perceptron()
