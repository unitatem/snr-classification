import logging

import numpy as np
from klepto.archives import dir_archive
from sklearn.cluster import KMeans

from src import config


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

    features_db = dir_archive(config.features_db_path, {}, serialized=True, cached=True)
    features_db.load()

    data = np.array([], dtype=np.float32)
    data.shape = (0, 128)
    for batch in features_db:
        data = np.concatenate((data, features_db[batch]))

    kmeans = find_clusters(data, 3)

    # print("labels:")
    # print(kmeans.labels_)
    # print("predict:")
    # print(kmeans.predict([[0, 0], [4, 4]]))
    print("centers:")
    print(kmeans.cluster_centers_)
