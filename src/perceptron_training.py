import logging
import h5py
from keras import callbacks
from keras import metrics
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from random import shuffle
from src import config
from src.sample_sequence import SampleSequence


def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


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
    output_layer_size = len(cluster_db)
    cluster_db.close()
    model.add(Dense(output_layer_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[top_1_accuracy,
                           top_5_accuracy])
    return model


def create_class_mapping():
    cluster_db = h5py.File(config.clusters_db_path, 'r')
    class_names = list(cluster_db.keys())
    cluster_db.close()
    sorted(class_names)
    mapping = {}
    for i, class_name in enumerate(class_names):
        mapping[class_name] = i
    return mapping


def divide_data():
    cluster_db = h5py.File(config.clusters_db_path)
    training_ids = []
    test_ids = []
    for class_name in cluster_db:
        class_ids = []
        for photo_name in cluster_db[class_name]:
            class_ids.append((class_name, photo_name))
        shuffle(class_ids)
        divide_point = int(len(class_ids) * config.training_total_ratio)
        training_ids += class_ids[:divide_point]
        test_ids += class_ids[divide_point:]
    cluster_db.close()
    shuffle(training_ids)
    shuffle(test_ids)

    return training_ids, test_ids


def get_labels(sample_ids):
    mapping = create_class_mapping()
    labels = [mapping[sample_id[0]] for sample_id in sample_ids]
    labels = to_categorical(labels, num_classes=len(mapping))
    return labels


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)
    training_ids, test_ids = divide_data()

    training_gen = SampleSequence(training_ids, get_labels(training_ids), config.batch_size)
    test_gen = SampleSequence(test_ids, get_labels(test_ids), config.batch_size)

    model = build_perceptron()
    callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required, 
                                       patience=config.max_no_improvement_epochs)
    model.fit_generator(training_gen, validation_data=test_gen, epochs=config.max_epochs, callbacks=[callback])

    training_gen.close()
    test_gen.close()

