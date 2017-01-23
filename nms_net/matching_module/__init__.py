
import os.path

import tensorflow as tf


__all__ = 'detection_matching'

this_path = os.path.dirname(os.path.realpath(__file__))
matching_module = tf.load_op_library(os.path.join(this_path, 'det_matching.so'))
tf.NotDifferentiable("DetectionMatching")

detection_matching = matching_module.detection_matching
