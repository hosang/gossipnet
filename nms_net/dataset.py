from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imread, imresize

from nms_net import cfg

DEBUG = False


def load_roi(need_images, roi, is_training=False):
    if DEBUG:
        print('loading ', roi)

    # make a copy so we don't keep the loaded image around
    roi = dict(roi)

    # if is_training:
    #     maxdet = cfg.train.max_num_detections
    #     if maxdet > 0 and maxdet < roi['det_classes'].size:
    #         sel = np.random.choice(roi['det_classes'].size, size=maxdet,
    #                                replace=False)
    #         roi['det_classes'] = roi['det_classes'][sel]
    #         roi['dets'] = roi['dets'][sel, :]
    #         roi['det_scores'] = roi['det_scores'][sel]

    if need_images:
        roi['image'], im_scale = load_image(roi['filename'], roi['flipped'])
        roi['image'] = roi['image'][None, ...]
        # don't do multiplications inplace
        if 'dets' in roi:
            roi['dets'] = roi['dets'] * im_scale
        if 'gt_boxes' in roi:
            roi['gt_boxes'] = roi['gt_boxes'] * im_scale
    return roi


def load_image(path, flipped):
    target_size = cfg.image_target_size
    max_size = cfg.image_max_size

    im = imread(path)
    if len(im.shape) == 2:
        im = np.tile(im[..., None], (1, 1, 3))
    h, w = im.shape[:2]
    im_size_min = min(h, w)
    im_size_max = max(h, w)
    im_scale = target_size / im_size_min
    if round(im_scale * im_size_max) > max_size:
        im_scale = max_size / im_size_max

    if flipped:
        im = im[:, ::-1, :].copy()
    im = imresize(im, im_scale)
    return im, im_scale


class Dataset(object):
    def __init__(self, imdb, batch_size, need_images):
        self._imdb = imdb
        self._roidb = imdb['roidb']
        self._batch_size = batch_size
        self._need_images = need_images
        self._shuffle()

    def _shuffle(self):
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        if DEBUG:
            print(self._perm)
        self._cur = 0

    def next_batch(self):
        if self._cur + self._batch_size >= self._perm.size:
            self._shuffle()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        assert len(db_inds) == 1
        roi = self._roidb[db_inds[0]]
        roi = load_roi(self._need_images, roi, is_training=True)
        return roi
