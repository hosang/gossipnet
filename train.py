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
    while not coord.should_stop():
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
    return coord, thread


def get_dataset():
    train_imdb = imdb.get_imdb(cfg.train.imdb)
    # TODO(jhosang): print stats of the data
    imdb.prepro_train(train_imdb)
    return Dataset(train_imdb, 1)


def train(device):
    np.random.seed(cfg.random_seed)
    with tf.device(device):
        dataset = get_dataset()
        preloaded_batch, enqueue_op, enqueue_placeholders, q_size = setup_preloading(
                Gnet.batch_spec)
        reg = tf.contrib.layers.l2_regularizer(cfg.train.weight_decay)
        net = Gnet(batch=preloaded_batch, weight_reg=reg, **cfg.gnet)
        lr_gen = LearningRate()
        reg_ops = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        learning_rate, train_op = get_optimizer(
                net.loss + tf.reduce_mean(reg_ops))

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        maintain_averages_op = ema.apply([net.loss])
        # update moving averages after every loss evaluation
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)
        average_loss = ema.average(net.loss)

    config = tf.ConfigProto(
            allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        coord, prefetch_thread = start_preloading(
            sess, enqueue_op, dataset, enqueue_placeholders)

        for it in range(1, cfg.train.num_iter + 1):
            _, loss = sess.run(
                [train_op, average_loss],
                feed_dict={learning_rate: lr_gen.get_lr(it)})

            if it % 20 == 0:
                print('iter {}   loss {}'.format(it, loss))

            # TODO(jhsoang): https://www.tensorflow.org/api_docs/python/summary/
            # TODO(jhosang): save snapshot
            # TODO(jhosang): eval run, produce mAP
            # TODO(jhosang): write best-model-symlink
    coord.request_stop()
    coord.join([prefetch_thread])


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
