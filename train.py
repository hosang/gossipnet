#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import threading

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import imdb
from nms_net import cfg
from nms_net.network import Gnet
from nms_net.config import cfg_from_file
from nms_net.dataset import Dataset


class LearningRate(object):
    def __init__(self):
        self.steps = cfg.train.lr_multi_step
        self.current_step = 0

    def get_lr(self, iter):
        lr = self.steps[self.current_step][1]
        if iter == self.steps[self.current_step][0]:
            self.current_step += 1
        return lr


def get_optimizer(loss_op):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    if cfg.train.optimizer == 'adam':
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif cfg.train.optimizer == 'sgd':
        opt_func = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=cfg.train.momentum)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.train.optimizer))
    if cfg.train.gradient_clipping > 0:
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), cfg.train.gradient_clipping)
        train_op = opt_func.apply_gradients(zip(grads, tvars))
    else:
        train_op = slim.learning.create_train_op(loss_op, opt_func)
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
    shapes = [shape for _, (_, shape) in spec]
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
    train_imdb = imdb.get_imdb(cfg.train.imdb)
    # TODO(jhosang): print stats of the data
    imdb.prepro_train(train_imdb)
    return Dataset(train_imdb, 1), train_imdb


def validation(net):
    valset = get_dataset(cfg.train.val_imdb, one_pass=True)
    all_labels = []
    all_scores = []
    num_objs = sum(np.sum(np.logical_not(roi['gt_crowd']))
                   for roi in val_imdb['roidb'])
    for i in range(len(valset['roidb'])):
        weights, labels, scores = sess.run(
                [net.weights, net.labels, net.predictions])
        # filter out ignored detections
        mask = weights > 0.0
        all_labels.append(labels[mask])
        all_scores.append(scores[mask])

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ord = np.argsort(-scores)
    scores = scores[ord]
    labels = labels[ord]
    fp = np.astype(np.cumsum(np.astype(labels == 0, dtype=np.int32)), dtype=np.float32)
    tp = np.astype(np.cumsum(np.astype(labels == 1, dtype=np.int32)), dtype=np.float32)
    recall = tp / num_objs
    precision = tp / (fp + tp)
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])


def train(device):
    # TODO(jhosang): implement training resuming
    np.random.seed(cfg.random_seed)
    # with tf.device(device):
    dataset, train_imdb = get_dataset()
    preloaded_batch, enqueue_op, enqueue_placeholders, q_size = setup_preloading(
            Gnet.batch_spec)
    reg = tf.contrib.layers.l2_regularizer(cfg.train.weight_decay)
    net = Gnet(batch=preloaded_batch, weight_reg=reg,
               random_seed=cfg.random_seed, **cfg.gnet)
    lr_gen = LearningRate()
    reg_ops = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_op = tf.reduce_mean(reg_ops)
    optimized_loss = net.loss + reg_op
    learning_rate, train_op = get_optimizer(optimized_loss)

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    per_det_loss = optimized_loss / train_imdb['avg_num_dets']
    per_det_data_loss = net.loss / train_imdb['avg_num_dets']
    per_det_loss.set_shape([])
    per_det_data_loss.set_shape([])
    maintain_averages_op = ema.apply([per_det_loss, per_det_data_loss])
    # update moving averages after every loss evaluation
    with tf.control_dependencies([train_op]):
        train_op = tf.group(maintain_averages_op)
    average_loss = ema.average(per_det_loss)
    average_data_loss = ema.average(per_det_data_loss)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', net.loss)
        tf.summary.scalar('loss_per_det', per_det_loss)
        tf.summary.scalar('data_loss_per_det', per_det_data_loss)
        tf.summary.scalar('regularizer', reg_op)
        tf.summary.scalar('lr', learning_rate)
        if tf.__version__.startswith('0.11'):
            merge_summaries_op = tf.merge_all_summaries()
        else:
            merge_summaries_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=None)
    config = tf.ConfigProto(
            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if tf.__version__.startswith('0.11'):
            train_writer = tf.train.SummaryWriter(cfg.log_dir, sess.graph)
            tf.initialize_all_variables().run()
        else:
            train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)
            tf.global_variables_initializer().run()
        coord = start_preloading(
            sess, enqueue_op, dataset, enqueue_placeholders)

        for it in range(1, cfg.train.num_iter + 1):
            if coord.should_stop():
                break

            _, avg_loss, avg_data_loss, reg_val, summary = sess.run(
                [train_op, average_loss, average_data_loss, reg_op,
                 merge_summaries_op],
                feed_dict={learning_rate: lr_gen.get_lr(it)})
            train_writer.add_summary(summary, it)

            if it % 20 == 0:
                print('iter {:6d}   lr {:8g}   loss {:8g} + {:8g} (reg) = {:8g}'.format(
                    it, lr_gen.get_lr(it), avg_data_loss, reg_val, avg_loss))

            if it % cfg.train.save_iter == 0 or it == cfg.train.num_iter:
                save_path = saver.save(sess, net.name, global_step=it)
                print('wrote model to {}'.format(save_path))

            # TODO(jhosang): eval run, produce mAP
            # TODO(jhosang): write best-model-symlink
        coord.request_stop()
        coord.join()
    print('training finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', nargs='?', type=bool, const=True,
                        default=False)
    parser.add_argument('-c', '--config', default='conf.yaml')
    args, unparsed = parser.parse_known_args()

    cfg_from_file(args.config)

    if args.cpu:
        device = '/cpu'
    else:
        device = '/gpu:{}'.format(args.gpu)

    train(device)


if __name__ == '__main__':
    main()
