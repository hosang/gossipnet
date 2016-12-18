from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Dataset(object):
    def __init__(self, imdb, batch_size):
        self._imdb = imdb
        self._roidb = imdb['roidb']
        self._batch_size = batch_size
        self._shuffle()

    def _shuffle(self):
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def next_batch(self):
        if self._cur + self._batch_size >= self._perm.size:
            self._shuffle()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        assert len(db_inds) == 1
        roi = self._roidb[db_inds[0]]
        return roi
