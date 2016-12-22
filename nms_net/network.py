
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from nms_net import cfg


this_path = os.path.dirname(os.path.realpath(__file__))
matching_module = tf.load_op_library(os.path.join(this_path, 'det_matching.so'))
tf.NotDifferentiable("DetectionMatching")

# TODO(jhosang): implement validation pass & mAP


def get_sample_weights(num_classes, labels):
    pos_weight = cfg.train.pos_weight
    with tf.variable_scope('loss_weighting'):
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


def get_resnet(image_tensor):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
        mean = tf.constant(cfg.pixel_mean, dtype=tf.float32,
                           shape=[1, 1, 1, 3], name='img_mean')
        image = image_tensor - mean
        if cfg.resnet_type == '50':
            net_fun = resnet_v1.resnet_v1_50
        elif cfg.resnet_type == '101':
            net_fun = resnet_v1.resnet_v1_101
        else:
            raise ValueError('unkown resnet type "{}"'.format(cfg.resnet_type))
        net, end_points = net_fun(image,
                                  global_pool=False, output_stride=16)
        layer_name = 'resnet_v1_{}/block2/unit_3/bottleneck_v1'.format(
            cfg.resnet_type)
        block2_out = end_points[layer_name]
        return block2_out


def enlarge_windows_convert_relative(boxdata, hw, padding=0.5):
    x1, y1, w, h, x2, y2, _ = boxdata
    cx = (x1 + x2 - 1) / 2.0
    cy = (y1 + y2 - 1) / 2.0
    nw2 = w * (0.5 + padding)
    nh2 = h * (0.5 + padding)

    # tensorflow wants relative coordinates
    new_yxyx = tf.concat(1, [cy - nh2, cx - nw2, cy + nh2 + 1, cx + nw2 + 1])
    denom = tf.expand_dims(tf.tile(tf.cast(hw, tf.float32), [2]), 0)
    new_yxyx = new_yxyx / denom
    return new_yxyx


def crop_windows(imfeats, boxdata):
    hw = tf.shape(imfeats)[1:3]
    rel_yxyx = enlarge_windows_convert_relative(boxdata, hw)
    n_boxes = tf.pack([tf.shape(boxdata[0])[0]])
    box_ind = tf.zeros(n_boxes, dtype=tf.int32)
    crop_size = tf.constant([cfg.imfeat_crop_height, cfg.imfeat_crop_width],
                            dtype=tf.int32)
    detection_feats = tf.image.crop_and_resize(
        imfeats, rel_yxyx, box_ind, crop_size, name='roi_pooling')
    return detection_feats


class Gnet(object):
    name = 'gnet'
    dets = None
    det_scores = None
    gt_boxes = None
    gt_crowd = None
    image = None

    @staticmethod
    def get_batch_spec():
        batch_spec = {
            'dets': (tf.float32, [None, 4]),
            'det_scores': (tf.float32, [None]),
            'gt_boxes': (tf.float32, [None, 4]),
            'gt_crowd': (tf.bool, [None]),
        }
        if cfg.gnet.imfeats:
            batch_spec['image'] = (tf.float32, [None, None, None, 3])
        return batch_spec

    def __init__(self, batch=None, weight_reg=None):
        # inputs
        if batch is None:
            for name, (dtype, shape) in self.get_batch_spec().items():
                setattr(self, name, tf.placeholder(dtype, shape=shape))
        else:
            for name, (dtype, shape) in self.get_batch_spec().items():
                batch[name].set_shape(shape)
                setattr(self, name, batch[name])

        if cfg.gnet.imfeats:
            self.imfeats = get_resnet(self.image)

        with tf.variable_scope('gnet'):
            with tf.variable_scope('preprocessing'):
                # generate useful box transformations (once)
                self.dets_boxdata = self._xyxy_to_boxdata(self.dets)
                self.gt_boxdata = self._xyxy_to_boxdata(self.gt_boxes)

                # overlaps
                self.det_anno_iou = self._iou(
                    self.dets_boxdata, self.gt_boxdata, self.gt_crowd)
                self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)

                # find neighbors
                self.neighbor_pair_idxs = tf.where(tf.greater_equal(
                    self.det_det_iou, cfg.gnet.neighbor_thresh))
                pair_c_idxs = self.neighbor_pair_idxs[:, 0]
                pair_n_idxs = self.neighbor_pair_idxs[:, 1]

                # generate handcrafted pairwise features
                pw_feats = self._geometry_feats(pair_c_idxs, pair_n_idxs)

            # initializers for network weights
            weights_init = tf.contrib.layers.xavier_initializer(
                seed=cfg.random_seed)
            biases_init = tf.constant_initializer()

            self.num_dets = tf.shape(self.dets)[0]

            if cfg.gnet.imfeats:
                self.det_imfeats = crop_windows(self.imfeats, self.dets_boxdata)
                self.det_imfeats = tf.contrib.layers.flatten(self.det_imfeats)
                with tf.variable_scope('reduce_imfeats'):
                    start_feat = tf.contrib.layers.fully_connected(
                        inputs=self.det_imfeats, num_outputs=cfg.gnet.shortcut_dim,
                        activation_fn=None,
                        weights_initializer=weights_init,
                        biases_initializer=biases_init)
            else:
                with tf.variable_scope('gnet'):
                    shortcut_shape = tf.pack([self.num_dets, cfg.gnet.shortcut_dim])
                    start_feat = tf.zeros(shortcut_shape, dtype=tf.float32)
            self.block_feats = []
            self.block_feats.append(start_feat)

            # stack the blocks
            for block_idx in range(1, cfg.gnet.num_blocks + 1):
                outfeats = self._block(
                    block_idx, self.block_feats[-1],
                    weights_init, biases_init, pair_c_idxs,
                    pair_n_idxs, pw_feats, weight_reg)
                self.block_feats.append(outfeats)

            # do prediction
            feats = self.block_feats[-1]
            for i in range(1, cfg.gnet.num_predict_fc):
                with tf.variable_scope('predict/fc{}'.format(i)):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=cfg.gnet.predict_fc_dim,
                        activation_fn=None,
                        weights_initializer=weights_init,
                        biases_initializer=biases_init)

            with tf.variable_scope('predict/logits'):
                prediction = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=1,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    biases_initializer=biases_init)
                self.prediction = tf.reshape(prediction, [-1])

            with tf.variable_scope('loss'):
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
                sample_class2 = tf.select(
                        tf.logical_and(self.det_gt_matching >= 0,
                                       tf.logical_not(det_crowd)),
                        tf.ones(tf.shape(sample_class)), sample_class)
                sample_weight = get_sample_weights(2, sample_class2)
                self.weights = self.weights * sample_weight

                # logistic_loss = weighted_logistic_loss(
                #     self.prediction, self.labels, self.weights)
                sample_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    self.prediction, self.labels)
                weighted_losses = sample_losses * self.weights
                self.loss = tf.reduce_sum(
                    weighted_losses, name='cls_loss')
                self.loss.set_shape([])
                tf.contrib.losses.add_loss(self.loss)

    @staticmethod
    def _block(block_idx, infeats, weights_init, biases_init,
               pair_c_idxs, pair_n_idxs, pw_feats, weight_reg):
        with tf.variable_scope('block{}'.format(block_idx)):
            feats = tf.contrib.layers.fully_connected(
                inputs=infeats, num_outputs=cfg.gnet.reduced_dim,
                activation_fn=tf.nn.relu,
                weights_initializer=weights_init,
                weights_regularizer=weight_reg,
                biases_initializer=biases_init,
                scope='reduce_dim')

            with tf.variable_scope('build_context'):
                c_feats = tf.gather(feats, pair_c_idxs)
                n_feats = tf.gather(feats, pair_n_idxs)

                # zero out features where c_idx == n_idx
                is_id_row = tf.equal(pair_c_idxs, pair_n_idxs)
                zeros = tf.zeros(tf.shape(n_feats), dtype=feats.dtype)
                n_feats = tf.select(is_id_row, zeros, n_feats)

                feats = tf.concat(1, [pw_feats, c_feats, n_feats])

            for i in range(1, cfg.gnet.num_block_pw_fc + 1):
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=cfg.gnet.pairfeat_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='pw_fc{}'.format(i))

            with tf.variable_scope('pooling'):
                feats = tf.segment_max(feats, pair_c_idxs, name='max')

            for i in range(1, cfg.gnet.num_block_fc):
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=cfg.gnet.pairfeat_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='fc{}'.format(i))

            feats = tf.contrib.layers.fully_connected(
                inputs=feats, num_outputs=cfg.gnet.shortcut_dim,
                activation_fn=None,
                weights_initializer=weights_init,
                weights_regularizer=weight_reg,
                biases_initializer=biases_init,
                scope='fc{}'.format(cfg.gnet.num_block_fc))

            with tf.variable_scope('shortcut'):
                outfeats = tf.nn.relu(infeats + feats)
        return outfeats

    def _geometry_feats(self, c_idxs, n_idxs):
        with tf.variable_scope('pairwise_features'):
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
    def _xyxy_to_boxdata(a):
        ax1 = tf.slice(a, [0, 0], [-1, 1])  # a[:, 0]
        ay1 = tf.slice(a, [0, 1], [-1, 1])  # a[:, 1]
        ax2 = tf.slice(a, [0, 2], [-1, 1])  # a[:, 2]
        ay2 = tf.slice(a, [0, 3], [-1, 1])  # a[:, 3]
        aw = ax2 - ax1
        ah = ay2 - ay1

        area = tf.mul(aw, ah)
        return (ax1, ay1, aw, ah, ax2, ay2, area)

    @staticmethod
    def _iou(a, b, crowd=None):
        with tf.variable_scope('iou'):
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
                return tf.select(crowd, ioa, iou)

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
