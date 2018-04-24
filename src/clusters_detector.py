import h5py
import logging
import numpy as np

from sklearn.cluster import KMeans

from src import config


def load_database(db_path):
    logging.info("Loading data {db}".format(db=db_path))
    features_db = h5py.File(db_path, "r")
    return features_db


def calculate_basic_record_cnt(db):
    return sum([db[class_name][photo_name].shape[0] for class_name in db for photo_name in db[class_name]])


def concatenate_data(features_db):
    logging.info("Concatenating data")
    number_of_records = calculate_basic_record_cnt(features_db)
    data = np.empty(shape=(number_of_records, 128), dtype=np.float32)
    idx = 0
    for class_name in features_db:
        for photo_name in features_db[class_name]:
            rows = features_db[class_name][photo_name].shape[0]
            data[idx:idx + rows, :] = features_db[class_name][photo_name]
            idx += rows
    return data


def find_clusters(data, clusters_cnt):
    """
    find data clusters
    :param data: hyperdimensional data, data has to have more samples than required number of clusters
    :param clusters_cnt: required number of clusters
    :return: model containing clusters parameters
    """
    logging.info("Finding clusters [{clusters_cnt}]".format(clusters_cnt=clusters_cnt))
    # fixed random_state for simpler and deterministic comparison between runs of algorithm
    kmeans = KMeans(n_clusters=clusters_cnt, random_state=0).fit(data)
    return kmeans


def clusterize_data_and_create_db(features_db, kmeans, clusters_db_path):
    logging.info("Changing space from features into [{clusters_db_path}]".format(clusters_db_path=clusters_db_path))
    cluster_db_file = h5py.File(clusters_db_path, "w")
    for class_name in features_db:
        grp = cluster_db_file.create_group(class_name)
        for photo_name in features_db[class_name]:
            # descriptors changed to labels from k-means
            labels = kmeans.predict(features_db[class_name][photo_name])
            bins = np.bincount(labels, minlength=clusters_cnt)
            grp.create_dataset(photo_name, data=bins)
    cluster_db_file.close()


def calculate_labels_dim(h5_file):
    """
    :param h5_file: hdf5 file as python dict
    :return: np. ndarray containing number of entries for each class
    """
    dim = np.empty(shape=len(h5_file))
    for i, class_name in enumerate(h5_file.keys()):
        dim[i] = len(h5_file[class_name])
    return dim


if __name__ == "__main__":
    logging.basicConfig(filename="clusters.log", level=logging.DEBUG)

    features_db = load_database(config.groups_db_path['training'])
    data = concatenate_data(features_db)
    features_db.close()
    for clusters_cnt in range(config.clusters_count_start, config.clusters_count_stop + 1, config.clusters_count_step):
        kmeans = find_clusters(data, clusters_cnt)
        for group in config.data_type:
            group_features_db = load_database(config.groups_db_path[group])
            clusterize_data_and_create_db(group_features_db, kmeans, config.get_clusters_db_path(group, clusters_cnt))
            group_features_db.close()
