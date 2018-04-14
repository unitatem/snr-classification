import logging

import numpy as np
from klepto.archives import dir_archive
from sklearn.cluster import KMeans

from src import config


def load_database(db_path):
    logging.info("Loading data")
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


if __name__ == "__main__":
    logging.basicConfig(filename="clusters.log", level=logging.DEBUG)

    print("Loading data")
    features_db = load_database(config.features_db_path)
    print("Concatenating data")
    data = concatenate_data(features_db)
    print("kMeans running")
    kmeans = find_clusters(data, config.clusters_count)

    # print("centers:")
    # print(kmeans.cluster_centers_)

    print("Changing space from features into clusters")
    cluster_db = dir_archive(config.clusters_db_path, {}, serialized=True, cached=False)
    for class_name in features_db:
        print(class_name)
        cluster_batch = {}
        for photo_name in features_db[class_name]:
            labels = kmeans.predict(features_db[class_name][photo_name])

            cluster_batch[photo_name] = np.zeros(shape=config.clusters_count, dtype=np.int16)
            for label in labels:
                cluster_batch[photo_name][label] += 1
        cluster_db[class_name] = cluster_batch
