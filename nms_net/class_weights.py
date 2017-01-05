
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nms_net import cfg
from imdb.tools import get_class_counts


def class_equal_weights(imdb):
    num_classes = imdb['num_classes']
    posweight = cfg.train.pos_weight
    # what we want the expectation of each class weight to be
    expected_class_weight = np.array(
            [1 - posweight] + [posweight / num_classes] * num_classes,
            dtype=np.float32)
    class_counts = get_class_counts(imdb)
    num_samples = np.sum(class_counts)

    class_weights = num_samples * expected_class_weight / class_counts
    return class_weights
