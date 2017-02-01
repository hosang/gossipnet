
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from nms_net import cfg
from nms_net.roi_pooling_layer import roi_pooling_op, roi_pooling_op_grad
from nms_net import matching_module


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
        #with tf.name_scope('summaries'):
        #    tf.summary.tensor_summary('exp_class_weights', exp_class_weights)
        #    tf.summary.tensor_summary('counts', class_counts)
        #    tf.summary.tensor_summary('class_weights', class_weights)
    return sample_weights


def weighted_logistic_loss(logits, labels, instance_weights):
    # http://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow
    # loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
    # so rescaling x, is weighting the loss
    return tf.nn.sigmoid_cross_entropy_with_logits(
            logits * instance_weights, labels)


def get_resnet(image_tensor):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
        mean = tf.constant(cfg.pixel_mean, dtype=tf.float32,
                           shape=[1, 1, 1, 3], name='img_mean')
        image = image_tensor - mean

        assert cfg.resnet_type == '101'
        net_fun = resnet_v1.resnet_v1_101
        net, end_points = net_fun(image,
                                  global_pool=False, output_stride=16)
        layer_name = 'resnet_v1_{}/block3/unit_22/bottleneck_v1'.format(
            cfg.resnet_type)
        block2_out = end_points[layer_name]
        stride = 16
        layer_prefixes = ['resnet_v1_101/conv1']
        layer_prefixes += ['resnet_v1_101/block1/unit_{}'.format(i + 1) for i in range(3)]
        layer_prefixes += ['resnet_v1_101/block2/unit_{}'.format(i + 1) for i in range(4)]
        layer_prefixes += ['resnet_v1_101/block3/unit_{}'.format(i + 1) for i in range(23)]
        layer_prefixes += ['resnet_v1_101/block4/unit_{}'.format(i + 1) for i in range(3)]
        ignore_prefixes = layer_prefixes[:cfg.gnet.freeze_n_imfeat_layers]
        idx = layer_prefixes.index('resnet_v1_101/block3/unit_22')
        ignore_prefixes += layer_prefixes[idx + 1:]
        return block2_out, stride, ignore_prefixes


def enlarge_windows(boxdata, padding=0.5):
    x1, y1, w, h, x2, y2, _ = boxdata
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw2 = w * (0.5 + padding)
    nh2 = h * (0.5 + padding)

    boxes = tf.concat(1, [cx - nw2, cy - nh2, cx + nw2, cy + nh2])
    return boxes


def to_tf_coords(boxes, hw):
    # tensorflow wants relative coordinates
    yxyx = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
    denom = tf.expand_dims(tf.tile(tf.cast(hw, tf.float32), [2]), 0)
    rel_yxyx = yxyx / denom
    return rel_yxyx


def to_frcn_coords(boxes):
    shape = tf.pack([tf.shape(boxes)[0], 1])
    new_boxes = tf.concat(1, [tf.zeros(shape, dtype=boxes.dtype), boxes])
    return new_boxes


def crop_windows(imfeats, boxdata, stride):
    boxes = enlarge_windows(boxdata)

    # hw = tf.shape(imfeats)[1:3]
    # n_boxes = tf.pack([tf.shape(boxdata[0])[0]])
    # box_ind = tf.zeros(n_boxes, dtype=tf.int32)
    # crop_size = tf.constant([cfg.imfeat_crop_height, cfg.imfeat_crop_width],
    #                         dtype=tf.int32)
    # rel_yxyx = to_tf_coords(boxes, hw)
    # detection_feats = tf.image.crop_and_resize(
    #     imfeats, rel_yxyx, box_ind, crop_size, name='roi_pooling')
    frcn_boxes = to_frcn_coords(boxes)
    detection_feats, _ = roi_pooling_op.roi_pool(
        imfeats, frcn_boxes, pooled_height=cfg.imfeat_crop_height,
        pooled_width=cfg.imfeat_crop_width, spatial_scale=1.0 / stride)
    return detection_feats


class Gnet(object):
    name = 'gnet'
    dets = None
    det_scores = None
    det_classes = None
    gt_boxes = None
    gt_crowd = None
    gt_classes = None
    image = None

    @staticmethod
    def get_batch_spec(num_classes):
        batch_spec = {
            'dets': (tf.float32, [None, 4]),
            'det_scores': (tf.float32, [None]),
            'gt_boxes': (tf.float32, [None, 4]),
            'gt_crowd': (tf.bool, [None]),
            'gt_classes': (tf.int32, [None]),
            'det_classes': (tf.int32, [None]),
        }
        if cfg.gnet.imfeats or cfg.gnet.load_imfeats:
            batch_spec['image'] = (tf.float32, [None, None, None, 3])
        return batch_spec

    def __init__(self, num_classes, class_weights=None, batch=None,
                 weight_reg=None):
        self.num_classes = num_classes
        self.multiclass = num_classes > 1

        # inputs
        if batch is None:
            for name, (dtype, shape) in self.get_batch_spec(num_classes).items():
                setattr(self, name, tf.placeholder(dtype, shape=shape))
        else:
            for name, (dtype, shape) in self.get_batch_spec(num_classes).items():
                batch[name].set_shape(shape)
                setattr(self, name, batch[name])

        self._ignore_prefixes = []
        if cfg.gnet.imfeats:
            self.imfeats, stride, self._ignore_prefixes = get_resnet(self.image)
            #self.imfeats = tf.Print(self.imfeats, [self.imfeats, tf.reduce_max(self.imfeats), tf.reduce_mean(self.imfeats)], summarize=20, message='imfeats')

        with tf.variable_scope('gnet'):
            with tf.variable_scope('preprocessing'):
                # generate useful box transformations (once)
                self.dets_boxdata = self._xyxy_to_boxdata(self.dets)
                self.gt_boxdata = self._xyxy_to_boxdata(self.gt_boxes)

                # overlaps
                self.det_anno_iou = self._iou(
                    self.dets_boxdata, self.gt_boxdata, self.gt_crowd)
                self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)
                if self.multiclass:
                    # set overlaps of detection and annotations to 0 if they
                    # have different classes, so they don't get matched in the
                    # loss
                    print('doing multiclass NMS')
                    same_class = tf.equal(
                        tf.reshape(self.det_classes, [-1, 1]),
                        tf.reshape(self.gt_classes, [1, -1]))
                    zeros = tf.zeros_like(self.det_anno_iou)
                    self.det_anno_iou = tf.select(same_class,
                                                  self.det_anno_iou, zeros)
                else:
                    print('doing single class NMS')

                # find neighbors
                self.neighbor_pair_idxs = tf.where(tf.greater_equal(
                    self.det_det_iou, cfg.gnet.neighbor_thresh))
                pair_c_idxs = self.neighbor_pair_idxs[:, 0]
                pair_n_idxs = self.neighbor_pair_idxs[:, 1]

                # generate handcrafted pairwise features
                self.num_dets = tf.shape(self.dets)[0]
                pw_feats = (self._geometry_feats(pair_c_idxs, pair_n_idxs)
                            * cfg.gnet.pw_feat_multiplyer)

            # initializers for network weights
            if cfg.gnet.weight_init == 'xavier':
                weights_init = tf.contrib.layers.xavier_initializer(
                    seed=cfg.random_seed)
            elif cfg.gnet.weight_init == 'caffe':
                weights_init =tf.contrib.layers.variance_scaling_initializer(
                    factor=1.0, mode='FAN_IN', uniform=True)
            elif cfg.gnet.weight_init == 'msra':
                weights_init =tf.contrib.layers.variance_scaling_initializer(
                    factor=2.0, mode='FAN_IN', uniform=False)
            else:
                raise ValueError('unknown weight init {}'.format(
                    cfg.gnet.weight_init))
            biases_init = tf.constant_initializer(cfg.gnet.bias_const_init)

            if cfg.gnet.num_pwfeat_fc > 0:
                with tf.variable_scope('pw_feats'):
                    pw_feats = self._pw_feats_fc(
                            pw_feats, weights_init, biases_init, weight_reg)
            self.pw_feats = pw_feats

            if cfg.gnet.imfeats:
                self.roifeats = crop_windows(
                        self.imfeats, self.dets_boxdata, stride)
                self.det_imfeats = tf.contrib.layers.flatten(self.roifeats)
                with tf.variable_scope('reduce_imfeats'):
                    if cfg.gnet.imfeat_dim > 0:
                        self.det_imfeats = tf.contrib.layers.fully_connected(
                            inputs=self.det_imfeats, num_outputs=cfg.gnet.imfeat_dim,
                            activation_fn=tf.nn.relu,
                            weights_initializer=weights_init,
                            weights_regularizer=weight_reg,
                            biases_initializer=biases_init)
                    start_feat = tf.contrib.layers.fully_connected(
                        inputs=self.det_imfeats, num_outputs=cfg.gnet.shortcut_dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        weights_regularizer=weight_reg,
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
                #outfeats = tf.Print(outfeats, [outfeats, tf.reduce_max(outfeats), tf.reduce_mean(outfeats)], summarize=20, message='block_{}'.format(block_idx))
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
                if class_weights is None:
                    class_weights = np.ones((num_classes + 1), dtype=np.float32)
                self.class_weights = tf.constant(class_weights, dtype=tf.float32)

                det_crowd = tf.cond(
                        tf.shape(self.gt_crowd)[0] > 0,
                        lambda: tf.gather(self.gt_crowd, tf.maximum(self.det_gt_matching, 0)),
                        lambda: tf.zeros(tf.shape(self.labels), dtype=tf.bool))
                det_class = tf.cond(
                        tf.shape(self.gt_crowd)[0] > 0,
                        lambda: tf.gather(tf.cast(self.gt_classes, tf.int32), tf.maximum(self.det_gt_matching, 0)),
                        lambda: tf.zeros(tf.shape(self.labels), dtype=tf.int32))
                det_class = tf.select(
                        tf.logical_and(self.det_gt_matching >= 0,
                                       tf.logical_not(det_crowd)),
                        det_class, tf.zeros_like(det_class))
                sample_weight = tf.gather(self.class_weights, det_class)
                self.weights = self.weights * sample_weight

                sample_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    self.prediction, self.labels)
                weighted_losses = sample_losses * self.weights
                self.loss_unnormed = tf.reduce_sum(
                    weighted_losses, name='cls_loss_unnormed')
                self.loss_unnormed.set_shape([])
                self.loss_normed = tf.reduce_mean(
                    weighted_losses, name='cls_loss_normed')
                self.loss_normed.set_shape([])
                if cfg.train.normalize_loss:
                    self.loss = self.loss_normed
                else:
                    self.loss = self.loss_unnormed
                tf.contrib.losses.add_loss(self.loss)

        # collect trainable variables
        tvars = tf.trainable_variables()
        self.trainable_variables = [
                var for var in tvars
                if (var.name.startswith('gnet') or var.name.startswith('resnet'))
                    and not any(var.name.startswith(pref)
                                for pref in self._ignore_prefixes)]

    @staticmethod
    def _pw_feats_fc(pw_feats, weights_init, biases_init, weight_reg):
        feats = pw_feats
        for i in range(1, cfg.gnet.num_pwfeat_fc):
            feats = tf.contrib.layers.fully_connected(
                inputs=feats, num_outputs=cfg.gnet.pwfeat_dim,
                activation_fn=tf.nn.relu,
                weights_initializer=weights_init,
                weights_regularizer=weight_reg,
                biases_initializer=biases_init,
                scope='fc{}'.format(i))
        feats = tf.contrib.layers.fully_connected(
            inputs=feats, num_outputs=cfg.gnet.pwfeat_narrow_dim,
            activation_fn=tf.nn.relu,
            weights_initializer=weights_init,
            weights_regularizer=weight_reg,
            biases_initializer=biases_init,
            scope='fc{}'.format(cfg.gnet.num_pwfeat_fc))
        return feats

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

            if cfg.gnet.neighbor_feats:
                neighbor_feats = tf.contrib.layers.fully_connected(
                    inputs=infeats, num_outputs=cfg.gnet.reduced_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='reduce_dim_neighbor')
            else:
                neighbor_feats = feats

            with tf.variable_scope('build_context'):
                c_feats = tf.gather(feats, pair_c_idxs)
                n_feats = tf.gather(neighbor_feats, pair_n_idxs)

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
            if self.multiclass:
                mc_score_shape = tf.pack([self.num_dets, self.num_classes])
                # classes are one-based (0 is background)
                mc_score_idxs = tf.stack(
                    [tf.range(self.num_dets), self.det_classes - 1], axis=1)
                det_scores = tf.scatter_nd(
                    mc_score_idxs, self.det_scores, mc_score_shape)
            else:
                det_scores = tf.expand_dims(self.det_scores, -1)
            c_score = tf.gather(det_scores, c_idxs)
            n_score = tf.gather(det_scores, n_idxs)
            tmp_ious = tf.expand_dims(self.det_det_iou, -1)
            ious = tf.gather_nd(tmp_ious, self.neighbor_pair_idxs)

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


class Knet(object):
    def __init__(self):
        with tf.variable_scope('knet'):
            with tf.variable_scope('preprocessing'):
                # generate useful box transformations (once)
                self.dets_boxdata = self._xyxy_to_boxdata(self.dets)
                self.gt_boxdata = self._xyxy_to_boxdata(self.gt_boxes)

                # overlaps
                self.det_anno_iou = self._iou(
                    self.dets_boxdata, self.gt_boxdata, self.gt_crowd)
                self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)
                if self.multiclass:
                    # set overlaps of detection and annotations to 0 if they
                    # have different classes, so they don't get matched in the
                    # loss
                    print('doing multiclass NMS')
                    same_class = tf.equal(
                        tf.reshape(self.det_classes, [-1, 1]),
                        tf.reshape(self.gt_classes, [1, -1]))
                    zeros = tf.zeros_like(self.det_anno_iou)
                    self.det_anno_iou = tf.select(same_class,
                                                  self.det_anno_iou, zeros)
                else:
                    print('doing single class NMS')

                # find neighbors
                self.neighbor_pair_idxs = tf.where(tf.greater_equal(
                    self.det_det_iou, cfg.gnet.neighbor_thresh))
                pair_c_idxs = self.neighbor_pair_idxs[:, 0]
                pair_n_idxs = self.neighbor_pair_idxs[:, 1]

                # generate handcrafted pairwise features
                self.num_dets = tf.shape(self.dets)[0]
                pw_feats = self._geometry_feats(pair_c_idxs, pair_n_idxs)

            # pw_feats -> K
            with tf.variable_scope('K'):
                num_fc = 3
                feats = pw_feats
                dim = cfg.knet.pairfeat_dim
                for i in range(1, num_fc + 1):
                    feats = tf.contrib.layers.fully_connected(
                        inputs=feats, num_outputs=dim,
                        activation_fn=tf.nn.relu,
                        weights_initializer=weights_init,
                        weights_regularizer=weight_reg,
                        biases_initializer=biases_init,
                        scope='fc{}'.format(i))
                dim = cfg.knet.feat_dim * cfg.knet.feat_dim
                feats = tf.contrib.layers.fully_connected(
                    inputs=feats, num_outputs=dim,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='fc'.format(num_fc + 1))
                K = tf.segment_mean(feats, pair_c_idxs, name='mean')

            # imfeats -> f(x)
            with tf.variable_scope('imfeats'):
                self.imfeats, stride, self._ignore_prefixes = get_resnet(self.image)
                self.det_imfeats = crop_windows(
                        self.imfeats, self.dets_boxdata, stride)
                self.det_imfeats = tf.contrib.layers.flatten(self.det_imfeats)

                feats = tf.contrib.layers.fully_connected(
                    inputs=self.det_imfeats, num_outputs=cfg.knet.feat_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='reduce')

            f_new = tf.mat_mul(K, feats)
            with tf.variable_scope('predict'):
                self.prediction = tf.contrib.layers.fully_connected(
                    inputs=f_new, num_outputs=1,
                    activation_fn=None,
                    weights_initializer=weights_init,
                    weights_regularizer=weight_reg,
                    biases_initializer=biases_init,
                    scope='reduce')

        with tf.variable_scope('loss'):
            self.loss()

        # collect trainable variables
        tvars = tf.trainable_variables()
        self.trainable_variables = [
                var for var in tvars
                if (var.name.startswith('gnet') or var.name.startswith('resnet'))
                    and not any(var.name.startswith(pref)
                                for pref in self._ignore_prefixes)]

    def loss(self):
        # matching loss
        self.labels, self.weights, self.det_gt_matching = \
            matching_module.detection_matching(
                self.det_anno_iou, self.prediction, self.gt_crowd)

        # class weighting
        if class_weights is None:
            class_weights = np.ones((num_classes + 1), dtype=np.float32)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

        det_crowd = tf.cond(
                tf.shape(self.gt_crowd)[0] > 0,
                lambda: tf.gather(self.gt_crowd, tf.maximum(self.det_gt_matching, 0)),
                lambda: tf.zeros(tf.shape(self.labels), dtype=tf.bool))
        det_class = tf.cond(
                tf.shape(self.gt_crowd)[0] > 0,
                lambda: tf.gather(tf.cast(self.gt_classes, tf.int32), tf.maximum(self.det_gt_matching, 0)),
                lambda: tf.zeros(tf.shape(self.labels), dtype=tf.int32))
        det_class = tf.select(
                tf.logical_and(self.det_gt_matching >= 0,
                               tf.logical_not(det_crowd)),
                det_class, tf.zeros_like(det_class))
        sample_weight = tf.gather(self.class_weights, det_class)
        self.weights = self.weights * sample_weight

        sample_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            self.prediction, self.labels)
        weighted_losses = sample_losses * self.weights
        self.loss_unnormed = tf.reduce_sum(
            weighted_losses, name='cls_loss_unnormed')
        self.loss_unnormed.set_shape([])
        self.loss_normed = tf.reduce_mean(
            weighted_losses, name='cls_loss_normed')
        self.loss_normed.set_shape([])
        if cfg.train.normalize_loss:
            self.loss = self.loss_normed
        else:
            self.loss = self.loss_unnormed
        tf.contrib.losses.add_loss(self.loss)

