
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf


this_path = os.path.dirname(os.path.realpath(__file__))
matching_module = tf.load_op_library(os.path.join(this_path, 'det_matching.so'))


# TODO(jhosang): implement validation pass & mAP


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
    batch_spec = {
        'dets': (tf.float32, [None, 4]),
        'det_scores': (tf.float32, [None, 1]),
        'annos': (tf.float32, [None, 4]),
        'crowd': (tf.bool, [None]),
    }

    def __init__(self, batch=None, **kwargs):
        self.params = GnetParams(**kwargs)
        params = self.params

        # inputs
        if batch is None:
            for name, (dtype, shape) in self.batch_spec.items():
                setattr(self, name, tf.placeholder(dtype, shape=shape))
        else:
            for name, _ in self.batch_spec.items():
                setattr(self, name, batch[name])

        # generate useful box transformations (once)
        self.dets_boxdata = self._xywh_to_boxdata(self.dets)
        self.annos_boxdata = self._xywh_to_boxdata(self.annos)

        # overlaps
        self.det_anno_iou = self._iou(
            self.dets_boxdata, self.annos_boxdata, self.crowd)
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
                pair_n_idxs, pw_feats)
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

        # matching loss
        self.labels, self.weights, self.det_gt_matching = \
            matching_module.detection_matching(
                self.det_anno_iou, self.prediction,
                tf.cast(self.crowd, tf.float32))
        self.loss = weighted_logistic_loss(
            self.prediction, self.labels, self.weights)

    @staticmethod
    def _block(block_idx, params, infeats, weights_init, biases_init,
               pair_c_idxs, pair_n_idxs, pw_feats):
        with tf.name_scope('block{}'.format(block_idx)):
            with tf.name_scope('reduce_dim'):
                feats = tf.contrib.layers.fully_connected(
                    inputs=infeats, num_outputs=params.reduced_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    biases_initializer=biases_init)

            with tf.name_scope('build_context'):
                c_feats = tf.gather(feats, pair_c_idxs)
                n_feats = tf.gather(feats, pair_n_idxs)
                feats = tf.concat(1, [pw_feats, c_feats, n_feats])

            for i in range(1, params.num_block_pw_fc + 1):
                with tf.name_scope('pw_fc{}'.format(i)):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=params.pairfeat_dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        biases_initializer=biases_init)

            with tf.name_scope('pooling'):
                # zero out features where c_idx == n_idx
                is_id_row = tf.equal(pair_c_idxs, pair_n_idxs)
                zeros = tf.zeros(tf.shape(feats), dtype=feats.dtype)
                feats = tf.where(is_id_row, zeros, feats)

                # before_pool_feats = tf.Variable(tf.zeros([1], dtype=feats.dtype),
                #                                trainable=False, validate_shape=False)
                # zero_idxs = tf.where(tf.equal(pair_c_idxs, pair_n_idxs))
                # zero_idxs = tf.reshape(zero_idxs, [-1])

                # zeros_shape = tf.pack([num_dets, tf.shape(feats)[1]])
                # zeros = tf.zeros(zeros_shape, dtype=feats.dtype)
                # feats = tf.assign(before_pool_feats, feats, validate_shape=False)
                # feats = tf.scatter_update(feats, zero_idxs, zeros)
                feats = tf.segment_max(feats, pair_c_idxs, name='max')

            for i in range(1, params.num_block_fc):
                with tf.name_scope('fc{}'.format(i)):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=params.pairfeat_dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        biases_initializer=biases_init)

            with tf.name_scope('fc{}'.format(block_idx, params.num_block_fc)):
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=params.shortcut_dim,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    biases_initializer=biases_init)
            outfeats = tf.nn.relu(infeats + feats)
        return outfeats

    def _geometry_feats(self, c_idxs, n_idxs):
        c_score = tf.gather(self.det_scores, c_idxs)
        n_score = tf.gather(self.det_scores, n_idxs)
        tmp_ious = tf.expand_dims(self.det_det_iou, -1)
        ious = tf.gather_nd(tmp_ious, self.neighbor_pair_idxs)
        # TODO(jhosang): implement the rest of the pairwise features
        all = tf.concat(1, [c_score, n_score, ious])
        return all

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