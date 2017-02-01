#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import threading
from pprint import pprint
from datetime import datetime
import os

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import imdb
from nms_net import cfg
from nms_net.network import Gnet
from nms_net.config import cfg_from_file
from nms_net.dataset import Dataset, load_roi
from nms_net.class_weights import class_equal_weights


class LearningRate(object):
    def __init__(self):
        self.steps = cfg.train.lr_multi_step
        self.current_step = 0

    def get_lr(self, iter):
        if self.current_step >= len(self.steps):
            return self.steps[-1][1]
        lr = self.steps[self.current_step][1]
        if iter == self.steps[self.current_step][0]:
            self.current_step += 1
        return lr


class ModelManager(object):
    def __init__(self):
        self.models = []

    def add(self, global_iter, ap, model_file):
        self.models.append((global_iter, ap, model_file))

    def print_summary(self):
        best_ap, best_model_file = max((ap, model_file) for _, ap, model_file in self.models)
        print('{:10s}  {:6s}'.format('Iteration', 'mAP'))
        for it, ap, model_file in self.models:
            if model_file == best_model_file:
                print('{:10d}  {:6.1f}  (best)'.format(it, ap))
            else:
                print('{:10d}  {:6.1f}'.format(it, ap))

    def write_link_to_best(self, link):
        best_ap, best_model_file = max((ap, model_file) for _, ap, model_file in self.models)
        print('writing symlink {} -> {}'.format(link, best_model_file))
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(best_model_file, link)


def get_optimizer(loss_op, tvars):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    if cfg.train.optimizer == 'adam':
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif cfg.train.optimizer == 'sgd':
        opt_func = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=cfg.train.momentum)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.train.optimizer))
    train_op = slim.learning.create_train_op(
            loss_op, opt_func,
            variables_to_train=tvars,
            clip_gradient_norm=cfg.train.gradient_clipping)
    return learning_rate, train_op


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    for _ in range(cfg.train.num_iter):
        if coord.should_stop():
            return
        batch = dataset.next_batch()
        food = {ph: batch[name] for (name, ph) in placeholders}
        sess.run(enqueue_op, feed_dict=food)


def setup_preloading(batch_spec):
    spec = list(batch_spec.items())
    dtypes = [dtype for _, (dtype, _) in spec]
    enqueue_placeholders = [(name, tf.placeholder(dtype, shape=shape))
                            for name, (dtype, shape) in spec]
    q = tf.FIFOQueue(cfg.prefetch_q_size, dtypes)
    enqueue_op = q.enqueue([ph for _, ph in enqueue_placeholders])
    dequeue_op = q.dequeue()
    q_size = q.size()
    preloaded_batch = {name: dequeue_op[i] for i, (name, _) in enumerate(spec)}
    return preloaded_batch, enqueue_op, enqueue_placeholders, q_size


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()
    thread = threading.Thread(
        target=load_and_enqueue,
        args=(sess, enqueue_op, coord, dataset, placeholders))
    thread.start()
    coord.register_thread(thread)
    return coord


def get_dataset():
    train_imdb = imdb.get_imdb(cfg.train.imdb, is_training=True)
    need_imfeats = cfg.gnet.imfeats or cfg.gnet.load_imfeats
    return Dataset(train_imdb, 1, need_imfeats), train_imdb


def val_run(sess, net, val_imdb):
    roidb = val_imdb['roidb']
    batch_spec = net.get_batch_spec(num_classes=val_imdb['num_classes'])
    need_image = 'image' in batch_spec

    all_labels = []
    all_scores = []
    all_classes = []
    for i, roi in enumerate(roidb):
        if 'dets' not in roi or roi['dets'].size == 0:
            continue
        roi = load_roi(need_image, roi)
        feed_dict = {getattr(net, name): roi[name]
                     for name in batch_spec.keys()}
        weights, labels, scores = sess.run(
                [net.weights, net.labels, net.prediction],
                feed_dict=feed_dict)
        # filter out ignored detections
        mask = weights > 0.0
        all_labels.append(labels[mask])
        all_scores.append(scores[mask])
        all_classes.append(roi['det_classes'][mask])

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    return compute_aps(scores, classes, labels, val_imdb)


def compute_aps(scores, classes, labels, val_imdb):
    ord = np.argsort(-scores)
    scores = scores[ord]
    labels = labels[ord]
    classes = classes[ord]

    num_objs = sum(np.sum(np.logical_not(roi['gt_crowd']))
                   for roi in val_imdb['roidb'])
    multiclass_ap = _compute_ap(scores, labels, num_objs)

    all_cls = np.unique(classes)
    print(all_cls)
    cls_ap = []
    for cls in all_cls:
        mask = classes == cls
        c_scores = scores[mask]
        c_labels = labels[mask]
        cls_gt = (np.logical_and(np.logical_not(roi['gt_crowd']),
                                 roi['gt_classes'] == cls)
                  for roi in val_imdb['roidb'])
        c_num_objs = sum(np.sum(is_cls_gt)
                         for is_cls_gt in cls_gt)
        cls_ap.append(_compute_ap(c_scores, c_labels, c_num_objs))

    mAP = np.mean(cls_ap)
    return mAP, multiclass_ap, cls_ap


def _compute_ap(scores, labels, num_objs):
    # computer recall & precision
    fp = np.cumsum((labels == 0).astype(dtype=np.int32)).astype(dtype=np.float32)
    tp = np.cumsum((labels == 1).astype(dtype=np.int32)).astype(dtype=np.float32)
    recall = tp / num_objs
    precision = tp / (fp + tp)
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate(([0], recall, [recall[-1], 2]), axis=0)
    precision = np.concatenate(([1], precision, [0, 0]), axis=0)
    # computer AP
    c_recall = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
    inds = np.searchsorted(recall, c_recall, side='left')
    c_precision = precision[inds]
    ap = np.average(c_precision) * 100
    return ap


def draw_rects(rects, dashed=False, color='red'):
    import matplotlib.pyplot as plt
    linestyle = 'dashed' if dashed else 'solid'
    for i in range(rects.shape[0]):
        rect = rects[i, :]
        plt.gca().add_patch(
            plt.Rectangle((rect[0], rect[1]), rect[2] - rect[0],
                          rect[3] - rect[1], fill=False,
                          edgecolor=color, linewidth=3,
                          linestyle=linestyle)
            )


def train(resume, visualize):
    np.random.seed(cfg.random_seed)
    dataset, train_imdb = get_dataset()
    do_val = len(cfg.train.val_imdb) > 0
    if do_val:
        val_imdb = imdb.get_imdb(cfg.train.val_imdb, is_training=False)

    class_weights = class_equal_weights(train_imdb)
    preloaded_batch, enqueue_op, enqueue_placeholders, q_size = setup_preloading(
            Gnet.get_batch_spec(train_imdb['num_classes']))
    reg = tf.contrib.layers.l2_regularizer(cfg.train.weight_decay)
    net = Gnet(num_classes=train_imdb['num_classes'], batch=preloaded_batch,
               weight_reg=reg, class_weights=class_weights)
    lr_gen = LearningRate()
    # reg_ops = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_op = tf.reduce_mean(reg_ops)
    # optimized_loss = net.loss + reg_op
    optimized_loss = tf.contrib.losses.get_total_loss()
    learning_rate, train_op = get_optimizer(optimized_loss, net.trainable_variables)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', net.loss)
        tf.summary.scalar('loss_normed', net.loss_normed)
        tf.summary.scalar('loss_unnormed', net.loss_unnormed)
        tf.summary.scalar('lr', learning_rate)
        tf.summary.scalar('q_size', q_size)
        if cfg.train.histograms:
            tf.summary.histogram('roi_feats', net.roifeats)
            tf.summary.histogram('det_imfeats', net.det_imfeats)
            tf.summary.histogram('pw_feats', net.pw_feats)
            for i, blockout in enumerate(net.block_feats):
                tf.summary.histogram('block{:02d}'.format(i + 1),
                                     blockout)
        merge_summaries_op = tf.summary.merge_all()

    with tf.name_scope('averaging'):
        ema = tf.train.ExponentialMovingAverage(decay=0.7)
        maintain_averages_op = ema.apply([net.loss_normed, net.loss_unnormed, optimized_loss])
        # update moving averages after every loss evaluation
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)
        smoothed_loss_normed = ema.average(net.loss_normed)
        smoothed_loss_unnormed = ema.average(net.loss_unnormed)
        smoothed_optimized_loss = ema.average(optimized_loss)

    if resume:
        ckpt = tf.train.get_checkpoint_state('./')
        restorer = tf.train.Saver()
    elif cfg.gnet.imfeats:
        variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        variables_to_exclude = \
            slim.get_variables_by_suffix('Adam_1', scope='resnet_v1') + \
            slim.get_variables_by_suffix('Adam', scope='resnet_v1') + \
            slim.get_variables_by_suffix('Momentum', scope='resnet_v1')
        restorer = tf.train.Saver(
            list(set(variables_to_restore) - set(variables_to_exclude)))

    saver = tf.train.Saver(max_to_keep=None)
    model_manager = ModelManager()
    config = tf.ConfigProto()
            #log_device_placement=True,
            #allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = start_preloading(
            sess, enqueue_op, dataset, enqueue_placeholders)

        start_iter = 1
        if resume:
            restorer.restore(sess, ckpt.model_checkpoint_path)
            start_iter = sess.run(
                    tf.get_default_graph().get_tensor_by_name("global_step:0")) + 1
        elif cfg.gnet.imfeats:
            restorer.restore(sess, cfg.train.pretrained_model)

        for it in range(start_iter, cfg.train.num_iter + 1):
            if coord.should_stop():
                break

            if visualize:
                # don't do actual training, just visualize data
                # TODO(jhosang): extract function for this
                import matplotlib.pyplot as plt
                tensors = {'train_op': train_op, 'image': net.image,
                        'gt_boxes': net.gt_boxes, 'gt_crowd': net.gt_crowd,
                        'dets': net.dets, 'det_matched': net.labels,
                        'det_matching': net.det_gt_matching,
                        'iou': net.det_anno_iou, 'scores': net.prediction,
                        }
                keys = tensors.keys()
                out = sess.run([tensors[k] for k in keys], feed_dict={learning_rate: lr_gen.get_lr(it)})
                res = dict(zip(keys, out))
                # import pickle
                # with open('batch{}.pkl'.format(it), 'wb') as fp:
                #     pickle.dump(res, fp)

                tp = res['det_matched'] > 0.5
                high_scoring = res['scores'] > 0.5
                max_iou = np.amax(res['iou'], axis=0)
                print('max_iou (per gt)', max_iou)
                print('argmax          ', np.argmax(res['iou'], axis=0))
                assignment_unique, assignment_count = np.unique(
                        res['det_matching'], return_counts=True)
                print('matching', assignment_unique, assignment_count)

                plt.subplot(2, 2, 1)
                im = res['image'][0, ...].astype(np.uint8)
                plt.imshow(im)
                draw_rects(res['gt_boxes'][res['gt_crowd'], :], dashed=True, color='blue')
                draw_rects(res['gt_boxes'][np.logical_not(res['gt_crowd']), :], color='blue')
                plt.subplot(2, 2, 2)
                plt.imshow(im)
                fp = np.logical_and(high_scoring, np.logical_not(tp))
                draw_rects(res['dets'][fp, :], color='red')
                draw_rects(res['dets'][tp, :], color='green')
                plt.subplot(2, 2, 3)
                plt.imshow(im)
                draw_rects(res['gt_boxes'][res['gt_crowd'], :], dashed=True, color='blue')
                draw_rects(res['gt_boxes'][np.logical_not(res['gt_crowd']), :], color='blue')
                draw_rects(res['dets'][tp, :], color='green')
                plt.show()
                continue

            _, val_total_loss, val_loss_normed, val_loss_unnormed, summary = sess.run(
                [train_op, smoothed_optimized_loss, smoothed_loss_normed,
                 smoothed_loss_unnormed, merge_summaries_op],
                feed_dict={learning_rate: lr_gen.get_lr(it)})
            train_writer.add_summary(summary, it)

            if it % cfg.train.display_iter == 0:
                print('{}  iter {:6d}   lr {:8g}   opt loss {:8g}     data loss normalized {:8g}   unnormalized {:8g}'.format(
                      datetime.now(), it, lr_gen.get_lr(it), val_total_loss, val_loss_normed, val_loss_unnormed))

            if do_val and it % cfg.train.val_iter == 0:
                print('{}  starting validation'.format(datetime.now()))
                val_map, mc_ap, pc_ap = val_run(sess, net, val_imdb)
                print('{}  iter {:6d}   validation pass:   mAP {:5.1f}   multiclass AP {:5.1f}'.format(
                      datetime.now(), it, val_map, mc_ap))

                save_path = saver.save(sess, net.name, global_step=it)
                print('wrote model to {}'.format(save_path))
                model_manager.add(it, val_map, save_path)
                model_manager.print_summary()
                model_manager.write_link_to_best('./gnet_best')

            elif it % cfg.train.save_iter == 0 or it == cfg.train.num_iter:
                save_path = saver.save(sess, net.name, global_step=it)
                print('wrote model to {}'.format(save_path))

        coord.request_stop()
        coord.join()
    print('training finished')
    if do_val:
        print('summary of validation performance')
        model_manager.print_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=False, action='store_true')
    parser.add_argument('-c', '--config', default='conf.yaml')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    args, unparsed = parser.parse_known_args()

    cfg_from_file(args.config)
    if args.visualize:
        cfg.gnet.load_imfeats = True
    pprint(cfg)

    train(args.resume, args.visualize)


if __name__ == '__main__':
    main()
