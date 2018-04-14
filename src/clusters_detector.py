import h5py
import logging
import numpy as np


from klepto.archives import dir_archive
from sklearn.cluster import KMeans

from src import config


def load_database(db_path):
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
    logging.info("Finding clusters [{clusters_cnt}]".format(clusters_cnt=clusters_count))
    # fixed random_state for simpler and deterministic comparison between runs of algorithm
    kmeans = KMeans(n_clusters=clusters_count, random_state=0).fit(data)
    return kmeans


def clusterize_data(features_db):
    cluster_db_file = h5py.File(config.clusters_db_path_h5py, "w")
    # cluster_db = dir_archive(config.clusters_db_path, {}, serialized=True, cached=False)
    for class_name in features_db:
        print(class_name)
        grp = cluster_db_file.create_group(class_name)
        # cluster_batch = {}
        for photo_name in features_db[class_name]:
            # zamienia deskryptory na labelki z k-means
            labels = kmeans.predict(features_db[class_name][photo_name])

        #     cluster_batch[photo_name] = np.zeros(shape=config.clusters_count, dtype=np.int16)
        #     cluster_batch[photo_name] = (np.bincount(labels))
        # cluster_db[class_name] = cluster_batch
        #     bins = np.zeros(shape=config.clusters_count, dtype=np.int16)
            bins = (np.bincount(labels))
            print("BINS: ", bins)
            grp.create_dataset(photo_name, data=bins)
    cluster_db_file.close()
        # cluster_db[class_name] = cluster_batch


def generate_labels(cluster_db):
    # ToDo: to be tested
    labels_file = h5py.File(config.labels_db_path, "w")
    labels_db = []
    for class_name in cluster_db:
        for photo_name in cluster_db[class_name]:
            labels_db.append(class_name)
    labels_file.create_dataset("labels", data=labels_db)
    labels_file.close()


if __name__ == "__main__":
    logging.basicConfig(filename="clusters.log", level=logging.DEBUG)

    print("Loading data")
    features_db = load_database(config.features_db_path)
    print("Concatenating data")
    data = concatenate_data(features_db)
    print("kMeans running")
    kmeans = find_clusters(data, config.clusters_count)

    print("Changing space from features into clusters")
    clusterize_data(features_db)

    # print(cluster_db)

