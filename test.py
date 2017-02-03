#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import tensorflow as tf

from tqdm import tqdm

import imdb
from nms_net import cfg, tools
from nms_net.config import cfg_from_file
from nms_net.dataset import load_roi
from nms_net.network import Gnet


def test_run(test_imdb):
    roidb = test_imdb['roidb']
    batch_spec = Gnet.get_batch_spec(num_classes=test_imdb['num_classes'])
    need_image = 'image' in batch_spec

    net = Gnet(num_classes=test_imdb['num_classes'])

    output_detections = []
    restorer = tf.train.Saver()
    config = tf.ConfigProto()
            #log_device_placement=True,
            #allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    forward_timer = tools.Timer()
    num_dets = num_images = 0
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        restorer.restore(sess, cfg.test_model)
        tf.report_uninitialized_variables()
        for i, roi in enumerate(tqdm(roidb)):
            if 'dets' not in roi or roi['dets'].size == 0:
                continue
            roi = load_roi(need_image, roi)
            feed_dict = {getattr(net, name): roi[name]
                         for name in batch_spec.keys()}
            forward_timer.tic()
            new_scores = sess.run(net.prediction, feed_dict=feed_dict)
            forward_timer.toc()

            num_dets += roi['dets'].shape[0]
            num_images += 1
            output_detections.append({
                'id': roi['id'],
                'dets': roi['dets'],
                'det_classes': roi['det_classes'],
                'det_scores': new_scores,
            })
    print('{:.6f}s per image with {:.1f} detections per image'.format(
        forward_timer.average_time, num_dets / num_images))
    return output_detections


def save_dets(testimdb, dets_as_dicts, output_file):
    """ Convert detection to the
    fast rcnn format (dets[num_classes][num_detections])
    and pickle them to disk.
    """
    cat_ids = [testimdb['class_to_cat_id'].get(cls, -1)
               for cls in testimdb['classes']]
    dets = [[] for _ in cat_ids]
    image_ids = []
    for det_dict in dets_as_dicts:
        image_ids.append(det_dict['id'])
        present_classes = set(np.unique(det_dict['det_classes']))
        for cls_ind, _ in enumerate(cat_ids):
            if cls_ind in present_classes:
                mask = det_dict['det_classes'] == cls_ind
                cls_dets = det_dict['dets'][mask, :]
                cls_scores = det_dict['det_scores'][mask]
                cls_dets = np.concatenate((cls_dets, cls_scores[:, None]),
                                          axis=1)
            else:
                cls_dets = np.zeros((0, 5), dtype=np.float32)
            dets[cls_ind]\
                .append(cls_dets)
    with open(output_file, 'wb') as fp:
        # python2.7 compativility
        pickle.dump((dets, image_ids, cat_ids), fp, protocol=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', help='detection file output')
    parser.add_argument('-c', '--config', default='conf.yaml')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-s', '--imdb', default=None)
    args, unparsed = parser.parse_known_args()

    cfg_from_file(args.config)

    if args.model is not None:
        cfg.test_model = args.model

    if args.imdb is not None:
        cfg.test.imdb = args.imdb

    test_imdb = imdb.get_imdb(cfg.test.imdb, is_training=False)
    dets = test_run(test_imdb)
    save_dets(test_imdb, dets, args.outfile)

if __name__ == '__main__':
    main()
