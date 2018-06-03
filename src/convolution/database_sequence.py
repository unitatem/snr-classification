from random import shuffle

import h5py
import numpy as np

from keras import utils
from keras.utils import to_categorical


class DatabaseSequence(utils.Sequence):

    def __init__(self, db_path, batch_size, total_cls_cnt, request_one_hot_labels):
        self.dataset = h5py.File(db_path, 'r')
        self.batch_size = batch_size
        self.total_cls_cnt = total_cls_cnt

        self.content = list()  # (cls, img)
        self._setup_content()
        self.request_one_hot_labels = request_one_hot_labels
        self.labels_hot = list()  # one hot matrix
        self.labels = list()
        self._setup_labels()

    def __len__(self):
        return int(np.ceil(self._length() / self.batch_size))

    def __getitem__(self, idx):
        batch_content = self.content[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.dataset[cls][img][0] for (cls, img) in batch_content]

        if self.request_one_hot_labels:
            batch_y = self.labels_hot[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def close(self):
        self.dataset.close()

    def _length(self):
        length = 0
        for cls in self.dataset:
            length += len(self.dataset[cls])
        return length

    def _setup_content(self):
        self.content = []
        for cls in self.dataset.keys():
            for img_hash in self.dataset[cls].keys():
                self.content.append((cls, img_hash))
        shuffle(self.content)

    # TODO: @mon3 might be improved
    def _setup_labels(self):
        cls_dict = dict()
        for i, key in enumerate(self.dataset.keys()):
            cls_dict[key] = i

        self.labels = []
        for i, (cls_name, _) in enumerate(self.content):
            self.labels.append(cls_dict[cls_name])

        self.labels_hot = to_categorical(self.labels, num_classes=self.total_cls_cnt)
