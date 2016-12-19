
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf

from nms_net import cfg


this_path = os.path.dirname(os.path.realpath(__file__))
matching_module = tf.load_op_library(os.path.join(this_path, 'det_matching.so'))
tf.NotDifferentiable("DetectionMatching")

# TODO(jhosang): implement validation pass & mAP

def get_sample_weights(num_classes, labels):
    pos_weight = cfg.train.pos_weight
    with tf.name_scope('loss_weighting'):
        exp_class_weights = tf.constant([1.0 - pos_weight] +
                [pos_weight / (num_classes - 1)] * (num_classes - 1),
                dtype=tf.float32)
        class_counts = tf.Variable(
                tf.ones([num_classes], dtype=tf.int64), trainable=False)

        t_label_count = tf.histogram_fixed_width(
                labels, [-0.5, num_classes - 0.5], nbins=num_classes,
                dtype=tf.int64)
        class_counts = class_counts.assign_add(t_label_count)
        total_samples = tf.cast(tf.reduce_sum(class_counts), dtype=tf.float32)
        class_weights = tf.truediv(
                tf.scalar_mul(total_samples, exp_class_weights),
                tf.cast(class_counts, tf.float32))
        int_labels = tf.cast(labels, tf.int32)
        sample_weights = tf.gather(class_weights, int_labels)
    return sample_weights


def weighted_logistic_loss(logits, labels, instance_weights):
    # http://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow
    # loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
    # so rescaling x, is weighting the loss
    return tf.nn.sigmoid_cross_entropy_with_logits(
            logits * instance_weights, labels)


class GnetParams(object):
    def __init__(self, neighbor_thresh=0.2, shortcut_dim=128, num_blocks=16,
                 reduced_dim=32, pairfeat_dim=None, gt_match_thresh=0.5,
                 num_block_pw_fc=2, num_block_fc=2, num_predict_fc=3,
                 block_dim=None, predict_fc_dim=None,
                 random_seed=42):
        self.neighbor_thresh = neighbor_thresh
        self.shortcut_dim = shortcut_dim
        self.num_blocks = num_blocks
        self.reduced_dim = reduced_dim
        self.pairfeat_dim = (reduced_dim * 2
                             if pairfeat_dim is None else pairfeat_dim)
        self.block_dim = (reduced_dim * 2
                          if block_dim is None else block_dim)
        self.predict_fc_dim = (shortcut_dim
                               if predict_fc_dim is None else predict_fc_dim)
        self.gt_match_thresh = gt_match_thresh
        self.num_block_pw_fc = num_block_pw_fc
        self.num_block_fc = num_block_fc
        self.num_predict_fc = num_predict_fc
        self.random_seed = random_seed


class Gnet(object):
    name = 'gnet'
    batch_spec = {
        'dets': (tf.float32, [None, 4]),
        'det_scores': (tf.float32, [None]),
        'gt_boxes': (tf.float32, [None, 4]),
        'gt_crowd': (tf.bool, [None]),
    }
    dets = None
    det_scores = None
    gt_boxes = None
    gt_crowd = None

    def __init__(self, batch=None, weight_reg=None, **kwargs):
        self.params = GnetParams(**kwargs)
        params = self.params

        # inputs
        if batch is None:
            for name, (dtype, shape) in self.batch_spec.items():
                setattr(self, name, tf.placeholder(dtype, shape=shape))
        else:
            for name, (dtype, shape) in self.batch_spec.items():
                batch[name].set_shape(shape)
                setattr(self, name, batch[name])

        # generate useful box transformations (once)
        self.dets_boxdata = self._xywh_to_boxdata(self.dets)
        self.gt_boxdata = self._xywh_to_boxdata(self.gt_boxes)

        # overlaps
        self.det_anno_iou = self._iou(
            self.dets_boxdata, self.gt_boxdata, self.gt_crowd)
        self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)

        # find neighbors
        self.neighbor_pair_idxs = tf.where(tf.greater_equal(
            self.det_det_iou, params.neighbor_thresh))
        pair_c_idxs = self.neighbor_pair_idxs[:, 0]
        pair_n_idxs = self.neighbor_pair_idxs[:, 1]

        # for now, start with zeros
        # TODO(jhosang): add image features here
        num_dets = tf.shape(self.dets)[0]
        shortcut_shape = tf.pack([num_dets, params.shortcut_dim])
        start_feat = tf.zeros(shortcut_shape, dtype=tf.float32)
        self.block_feats = []
        self.block_feats.append(start_feat)

        # generate handcrafted pairwise features
        pw_feats = self._geometry_feats(pair_c_idxs, pair_n_idxs)

        # initializers for network weights
        weights_init = tf.contrib.layers.xavier_initializer(
            seed=params.random_seed)
        biases_init = tf.constant_initializer()

        # stack the blocks
        for block_idx in range(1, params.num_blocks + 1):
            outfeats = self._block(
                block_idx, params, self.block_feats[-1],
                weights_init, biases_init, pair_c_idxs,
                pair_n_idxs, pw_feats, weight_reg)
            self.block_feats.append(outfeats)

        # do prediction
        feats = self.block_feats[-1]
        for i in range(1, params.num_predict_fc):
            with tf.name_scope('predict/fc{}'.format(i)):
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=params.predict_fc_dim,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    biases_initializer=biases_init)

        with tf.name_scope('predict/logits'):
            prediction = tf.contrib.layers.fully_connected(
                inputs=feats, num_outputs=1,
                activation_fn=None,
                weights_initializer=weights_init,
                biases_initializer=biases_init)
            self.prediction = tf.reshape(prediction, [-1])

        with tf.name_scope('loss'):
            # matching loss
            self.labels, self.weights, self.det_gt_matching = \
                matching_module.detection_matching(
                    self.det_anno_iou, self.prediction, self.gt_crowd)

            # class weighting
            sample_class = tf.zeros(tf.shape(self.labels), dtype=tf.float32)
            det_crowd = tf.cond(
                    tf.shape(self.gt_crowd)[0] > 0,
                    lambda: tf.gather(self.gt_crowd, tf.maximum(self.det_gt_matching, 0)),
                    lambda: tf.zeros(tf.shape(sample_class), dtype=tf.bool))
            sample_class2 = tf.where(
                    tf.logical_and(self.det_gt_matching >= 0,
                                   tf.logical_not(det_crowd)),
                    tf.ones(tf.shape(sample_class)), sample_class)
            sample_weight = get_sample_weights(2, sample_class2)
            self.weights = self.weights * sample_weight

            logistic_loss = weighted_logistic_loss(
                self.prediction, self.labels, self.weights)
            self.loss = tf.reduce_sum(logistic_loss)

    @staticmethod
    def _block(block_idx, params, infeats, weights_init, biases_init,
               pair_c_idxs, pair_n_idxs, pw_feats, weight_reg):
        with tf.name_scope('block{}'.format(block_idx)):
            with tf.name_scope('reduce_dim'):
                feats = tf.contrib.layers.fully_connected(
                    inputs=infeats, num_outputs=params.reduced_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init)

            with tf.name_scope('build_context'):
                c_feats = tf.gather(feats, pair_c_idxs)
                n_feats = tf.gather(feats, pair_n_idxs)

                # zero out features where c_idx == n_idx
                is_id_row = tf.equal(pair_c_idxs, pair_n_idxs)
                zeros = tf.zeros(tf.shape(n_feats), dtype=feats.dtype)
                n_feats = tf.where(is_id_row, zeros, n_feats)

                feats = tf.concat(1, [pw_feats, c_feats, n_feats])

            for i in range(1, params.num_block_pw_fc + 1):
                with tf.name_scope('pw_fc{}'.format(i)):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=params.pairfeat_dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        weights_regularizer=weight_reg,
                        biases_initializer=biases_init)

            with tf.name_scope('pooling'):
                feats = tf.segment_max(feats, pair_c_idxs, name='max')

            for i in range(1, params.num_block_fc):
                with tf.name_scope('fc{}'.format(i)):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=params.pairfeat_dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        weights_regularizer=weight_reg,
                        biases_initializer=biases_init)

            with tf.name_scope('fc{}'.format(block_idx, params.num_block_fc)):
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=params.shortcut_dim,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init)
            outfeats = tf.nn.relu(infeats + feats)
        return outfeats

    def _geometry_feats(self, c_idxs, n_idxs):
        det_scores = tf.expand_dims(self.det_scores, -1)
        c_score = tf.gather(det_scores, c_idxs)
        n_score = tf.gather(det_scores, n_idxs)
        tmp_ious = tf.expand_dims(self.det_det_iou, -1)
        ious = tf.gather_nd(tmp_ious, self.neighbor_pair_idxs)

        # TODO(jhosang): implement the rest of the pairwise features
        x1, y1, w, h, _, _, _ = self.dets_boxdata
        c_w = tf.gather(w, c_idxs)
        c_h = tf.gather(h, c_idxs)
        c_scale = (c_w + c_h) / 2.0
        c_cx = tf.gather(x1, c_idxs) + c_w / 2.0
        c_cy = tf.gather(y1, c_idxs) + c_h / 2.0

        n_w = tf.gather(w, n_idxs)
        n_h = tf.gather(h, n_idxs)
        n_cx = tf.gather(x1, n_idxs) + n_w / 2.0
        n_cy = tf.gather(y1, n_idxs) + n_h / 2.0

        # normalized x, y distance
        x_dist = (n_cx - c_cx)
        y_dist = (n_cy - c_cy)
        l2_dist = tf.sqrt(x_dist ** 2 + y_dist ** 2) / c_scale
        x_dist /= c_scale
        y_dist /= c_scale

        # scale difference
        log2 = tf.constant(np.log(2.0), dtype=tf.float32)
        w_diff = tf.log(n_w / c_w) / log2
        h_diff = tf.log(n_h / c_h) / log2
        aspect_diff = (tf.log(n_w / n_h) - tf.log(c_w / c_h)) / log2

        all = tf.concat(1, [c_score, n_score, ious, x_dist, y_dist, l2_dist,
            w_diff, h_diff, aspect_diff])
        return tf.stop_gradient(all)

    @staticmethod
    def _zero_diagonal(a):
        diag_shape = tf.shape(a)[0:1]
        z = tf.zeros(diag_shape, dtype=a.dtype)
        return tf.matrix_set_diag(a, z)

    @staticmethod
    def _xywh_to_boxdata(a):
        ax1 = tf.slice(a, [0, 0], [-1, 1])
        ay1 = tf.slice(a, [0, 1], [-1, 1])
        aw = tf.slice(a, [0, 2], [-1, 1])
        ah = tf.slice(a, [0, 3], [-1, 1])
        ax2 = tf.add(ax1, aw)
        ay2 = tf.add(ay1, ah)
        area = tf.mul(aw, ah)
        return (ax1, ay1, aw, ah, ax2, ay2, area)

    @staticmethod
    def _iou(a, b, crowd=None):
        a_area = tf.reshape(a[6], [-1, 1])
        b_area = tf.reshape(b[6], [1, -1])
        intersection = Gnet._intersection(a, b)
        union = tf.sub(tf.add(a_area, b_area), intersection)
        iou = tf.div(intersection, union)
        if crowd is None:
            return iou
        else:
            ioa = tf.div(intersection, a_area)
            crowd_multiple = tf.pack([tf.shape(a_area)[0], 1])
            crowd = tf.tile(tf.reshape(crowd, [1, -1]), crowd_multiple, name='det_gt_crowd')
            return tf.where(crowd, ioa, iou)

    @staticmethod
    def _intersection(a, b):
        """ Compute intersection between all ways of boxes in a and b.
        """
        ax1 = a[0]
        ay1 = a[1]
        ax2 = a[4]
        ay2 = a[5]

        bx1 = b[0]
        by1 = b[1]
        bx2 = b[4]
        by2 = b[5]

        x1 = tf.maximum(tf.reshape(ax1, [-1, 1]), tf.reshape(bx1, [1, -1]))
        y1 = tf.maximum(tf.reshape(ay1, [-1, 1]), tf.reshape(by1, [1, -1]))
        x2 = tf.minimum(tf.reshape(ax2, [-1, 1]), tf.reshape(bx2, [1, -1]))
        y2 = tf.minimum(tf.reshape(ay2, [-1, 1]), tf.reshape(by2, [1, -1]))
        w = tf.maximum(0.0, tf.sub(x2, x1))
        h = tf.maximum(0.0, tf.sub(y2, y1))
        intersection = tf.mul(w, h)
        return intersection
