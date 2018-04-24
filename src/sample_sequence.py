import h5py
from src import config
import numpy as np
from keras import utils


class SampleSequence(utils.Sequence):
    def __init__(self, sample_ids, sample_classes, batch_size, clusters_cnt):
        self.ids, self.classes = sample_ids, sample_classes
        self.batch_size = batch_size
        self.cluster_db = h5py.File(config.get_clusters_db_path(clusters_cnt), 'r')

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.classes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.cluster_db[x[0]][x[1]] for x in batch_ids]
        return np.array(batch_x), np.array(batch_y)

    def close(self):
        self.cluster_db.close()
