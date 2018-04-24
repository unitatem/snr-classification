import logging
import h5py
from keras import callbacks
from keras import metrics
from keras.callbacks import CSVLogger
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


def build_perceptron(clusters_cnt, layer_cnt, size_of_layers, activation):
    logging.info('Building perceptron: [{clusters_cnt} clusters_cnt, '
                 '{layer_cnt} layers, '
                 '{size_of_layers} neurons_in_every_layer, '
                 '{activation} activation]'
                 .format(clusters_cnt=clusters_cnt,
                         layer_cnt=layer_cnt,
                         size_of_layers=size_of_layers,
                         activation=activation))
    model = Sequential()
    for i in range(layer_cnt):
        if i == 0:
            model.add(Dense(layer_size, input_dim=clusters_cnt))
        else:
            model.add(Dense(layer_size))
        model.add(Activation(activation))
    cluster_db = h5py.File(config.get_clusters_db_path('training', clusters_cnt), 'r')
    output_layer_size = len(cluster_db)
    cluster_db.close()
    model.add(Dense(output_layer_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[top_1_accuracy,
                           top_5_accuracy])
    return model


def create_class_mapping(clusters_cnt):
    cluster_db = h5py.File(config.get_clusters_db_path('training', clusters_cnt), 'r')
    class_names = list(cluster_db.keys())
    cluster_db.close()
    sorted(class_names)
    mapping = {}
    for i, class_name in enumerate(class_names):
        mapping[class_name] = i
    return mapping



    # cluster_db = h5py.File(config.get_clusters_db_path(clusters_cnt))
def get_ids(cluster_db):
    ids = []
    for class_name in cluster_db:
        for photo_name in cluster_db[class_name]:
            ids.append((class_name, photo_name))
    shuffle(ids)
    return ids


def get_labels(sample_ids):
    mapping = create_class_mapping(clusters_cnt)
    labels = [mapping[sample_id[0]] for sample_id in sample_ids]
    labels = to_categorical(labels, num_classes=len(mapping))
    return labels


if __name__ == '__main__':
    logging.basicConfig(filename='perceptron.log', level=logging.DEBUG)

    stop_callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required,
                                            patience=config.max_no_improvement_epochs)
    for clusters_cnt in range(config.clusters_count_start, config.clusters_count_stop + 1, config.clusters_count_step):
        gens = {}
        for group in config.data_type:
            cluster_db = h5py.File(config.get_clusters_db_path(group, clusters_cnt), 'r')
            ids = get_ids(cluster_db)
            cluster_db.close()
            gens[group] = SampleSequence(ids, get_labels(ids),
                                         config.get_clusters_db_path(group, clusters_cnt), config.batch_size)
        for layer_cnt in range(config.layer_cnt_start, config.layer_cnt_stop + 1, config.layer_cnt_step):
            for layer_size in range(config.layer_size_start, config.layer_size_stop + 1, config.layer_size_step):
                for activation in config.activation_functions:
                    csv_logger = CSVLogger("perceptron_"
                                           + str(clusters_cnt) + "_"
                                           + str(layer_cnt) + "_"
                                           + str(layer_size) + "_"
                                           + activation + ".csv", append=True, separator=';')

                    model = build_perceptron(clusters_cnt, layer_cnt, layer_size, activation)
                    model.fit_generator(gens['training'],
                                        validation_data=gens['validation'],
                                        epochs=config.max_epochs,
                                        callbacks=[stop_callback, csv_logger])
                    # evaluate the model
                    scores = model.evaluate_generator(gens['test'])
                    for i in range(len(scores)):
                        logging.info("{name}: {value}"
                                     .format(name=model.metrics_names[i],
                                             value=scores[i]))

        for key in gens.keys():
            gens[key].close()
