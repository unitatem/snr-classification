import h5py
import logging
import numpy as np


from klepto.archives import dir_archive
from sklearn.cluster import KMeans

from src import config


def load_database(db_path):
    logging.debug("Loading data")
    features_db = dir_archive(db_path, {}, serialized=True, cached=True)
    features_db.load()
    return features_db


def concatenate_data(original):
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
    logging.debug("kMeans running")
    logging.info("Finding clusters [{clusters_cnt}]".format(clusters_cnt=clusters_count))
    # fixed random_state for simpler and deterministic comparison between runs of algorithm
    kmeans = KMeans(n_clusters=clusters_count, random_state=0).fit(data)
    return kmeans


def clusterize_data(features_db):
    logging.debug("Changing space from features into clusters")
    cluster_db_file = h5py.File(config.clusters_db_path_h5py, "w")
    for class_name in features_db:
        grp = cluster_db_file.create_group(class_name)
        for photo_name in features_db[class_name]:
            # descriptors changed to labels from k-means
            labels = kmeans.predict(features_db[class_name][photo_name])
            bins = np.bincount(labels)
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


def generate_labels():
    """

    :return: creates hdf5 file containing labels, prepared for the input of the multilayer percpetron
    """
    logging.debug("Generating labels...")
    cluster_db_file = h5py.File(config.clusters_db_path_h5py, "r")
    cluster_db_file = dict(cluster_db_file)
    labels_nr = calculate_labels_dim(cluster_db_file)
    # cluster_db_file.keys(): class folders
    # To retrieve the data with binned photo descriptors: np.array(file['0397']['1d7319df80044e29a04fa9b8c6456726'])
    labels_db = np.empty(shape=int(labels_nr.sum()))
    labels_idx = np.cumsum(labels_nr)
    labels_idx = labels_idx.astype(int)
    file_keys = list(cluster_db_file.keys())
    for i, high_idx in enumerate(labels_idx):
        low_index = 0
        if i != 0:
            low_index = labels_idx[i-1]
        labels_db[low_index:high_idx] = str(cluster_db_file[file_keys[i]].name)[2:]
    labels_file = h5py.File(config.labels_db_path_h5py, "w")
    labels_file.create_dataset("labels", data=labels_db)
    labels_file.close()
    logging.info("Labels generation finished.")


if __name__ == "__main__":
    logging.basicConfig(filename="clusters.log", level=logging.DEBUG)

    features_db = load_database(config.features_db_path)
    data = concatenate_data(features_db)
    kmeans = find_clusters(data, config.clusters_count)

    clusterize_data(features_db)

    generate_labels()
