import h5py
import logging
import numpy as np

from sklearn.cluster import KMeans

from src import config


def load_database(db_path):
    logging.info("Loading data {db}".format(db=db_path))
    features_db = h5py.File(config.features_db_path, "r")
    return features_db


def concatenate_data(features_db):
    logging.info("Concatenating data")
    data = np.array([], dtype=np.float32)
    data.shape = (0, 128)
    for class_name in features_db:
        for photo_name in features_db[class_name]:
            data = np.concatenate((data, features_db[class_name][photo_name]))
    return data


def find_clusters(data, clusters_count):
    """
    find data clusters
    :param data: hyperdimensional data, data has to have more samples than required number of clusters
    :param clusters_count: required number of clusters
    :return: model containing clusters parameters
    """
    logging.info("Finding clusters [{clusters_cnt}]".format(clusters_cnt=clusters_count))
    # fixed random_state for simpler and deterministic comparison between runs of algorithm
    kmeans = KMeans(n_clusters=clusters_count, random_state=0).fit(data)
    return kmeans


def clusterize_data(features_db, kmeans):
    logging.info("Changing space from features into clusters")
    cluster_db_file = h5py.File(config.clusters_db_path, "w")
    for class_name in features_db:
        grp = cluster_db_file.create_group(class_name)
        for photo_name in features_db[class_name]:
            # descriptors changed to labels from k-means
            labels = kmeans.predict(features_db[class_name][photo_name])
            bins = np.bincount(labels, minlength=config.clusters_count)
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

    features_db = load_database(config.features_db_path)
    data = concatenate_data(features_db)
    kmeans = find_clusters(data, config.clusters_count)
    clusterize_data(features_db, kmeans)
    features_db.close()
