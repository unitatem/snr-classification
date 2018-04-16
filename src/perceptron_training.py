import logging
from keras.models import Sequential
from keras.layers import Dense, Activation
from src import config


def build_perceptron():
    logging.info('Building perceptron: [' + ' '.join([str(x) for x in config.sizes_of_layers]) + ']')
    model = Sequential()
    for i, layer_size in enumerate(config.sizes_of_layers):
        if i == 0:
            model.add(Dense(layer_size, input_dim=config.clusters_count))
        else:
            model.add(Dense(layer_size))
    return model


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)
    model = build_perceptron()
