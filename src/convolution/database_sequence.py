import h5py
import numpy as np

from keras import utils
from keras.utils import to_categorical


class DatabaseSequence(utils.Sequence):
    def __init__(self, db_path, batch_size):
        self.dataset = h5py.File(db_path, 'r')
        self.batch_size = batch_size

        self.labels = list()
        self.key = [('cls', 'img') for _ in range(self._length())]
        idx = 0
        for i, cls in enumerate(self.dataset):
            for img in self.dataset[cls]:
                self.key[idx] = (cls, img)
                self.labels.append(i)
                idx += 1

        self.labels = to_categorical(self.labels, num_classes=50)

    def __len__(self):
        return int(np.ceil(self._length() / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.key[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.dataset[cls][img][0] for (cls, img) in batch_ids]

        return np.array(batch_x), np.array(batch_y)

    def _length(self):
        length = 0
        for cls in self.dataset:
            length += len(self.dataset[cls])
        return length

    def close(self):
        self.dataset.close()
