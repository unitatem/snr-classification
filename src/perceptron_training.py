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
    cluster_db = h5py.File(config.clusters_groups_db_path['training'], 'r')
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
    cluster_db = h5py.File(config.clusters_groups_db_path['training'], 'r')
    class_names = list(cluster_db.keys())
    cluster_db.close()
    sorted(class_names)
    mapping = {}
    for i, class_name in enumerate(class_names):
        mapping[class_name] = i
    return mapping


def get_ids(cluster_db):
    ids = []
    for class_name in cluster_db:
        for photo_name in cluster_db[class_name]:
            ids.append((class_name, photo_name))
    shuffle(ids)
    return ids


def get_labels(sample_ids):
    mapping = create_class_mapping()
    labels = [mapping[sample_id[0]] for sample_id in sample_ids]
    labels = to_categorical(labels, num_classes=len(mapping))
    return labels


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)

    ids = {}
    gens = {}
    for group in config.clusters_groups_db_path.keys():
        cluster_db = h5py.File(config.clusters_groups_db_path[group], 'r')
        ids = get_ids(cluster_db)
        cluster_db.close()
        gens[group] = SampleSequence(ids, get_labels(ids),
                                     config.clusters_groups_db_path[group], config.batch_size)

    model = build_perceptron()
    callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required,
                                       patience=config.max_no_improvement_epochs)
    model.fit_generator(gens['training'], validation_data=gens['validation'],
                        epochs=config.max_epochs, callbacks=[callback])
    model.evaluate_generator(gens['test'])

    # for i in range(len(gens['training'])):
    #     print(list(gens['training'][i][1]))

    for key in gens.keys():
        gens[key].close()
