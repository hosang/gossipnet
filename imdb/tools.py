from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def append_flipped(roidb):
    def flip(boxes, width):
        oldx1 = boxes[:, 0]
        oldx2 = boxes[:, 2]
        boxes[:, 0] = width - oldx2
        boxes[:, 2] = width - oldx1
        return boxes

    flipped = []
    for roi in roidb:
        width = roi['width']

        froi = dict(roi)
        froi['flipped'] = True
        if 'dets' in roi:
            froi['dets'] = flip(roi['dets'].copy(), width)
        if 'gt_boxes' in roi:
            froi['gt_boxes'] = flip(roi['gt_boxes'].copy(), width)
    return roidb + flipped


def drop_no_dets(roidb):
    res = [roi for roi in roidb if 'dets' in roi]
    return res


def drop_no_gt(roidb):
    res = [roi for roi in roidb if 'gt_boxes' in roi]
    return res


def only_keep_class(imdb, class_name):
    class_to_ind = imdb['class_to_ind']
    cls_ind = class_to_ind[class_name]

    imdb['classes'] = (imdb['classes'][0], imdb['classes'][cls_ind])
    imdb['class_to_ind'] = {cl: ind for ind, cl in enumerate(imdb['classes'])}

    roidb = imdb['roidb']
    for roi in roidb:
        if 'gt_classes' in roi:
            mask = roi['gt_classes'] == cls_ind
            roi['gt_classes'] = roi['gt_classes'][mask].copy()
            roi['gt_boxes'] = roi['gt_boxes'][mask].copy()
            roi['gt_crowd'] = roi['gt_crowd'][mask].copy()

        if 'det_classes' in roi:
            mask = roi['det_classes'] == cls_ind
            roi['det_classes'] = roi['det_classes'][mask].copy()
            roi['dets'] = roi['dets'][mask].copy()
            roi['det_scores'] = roi['det_scores'][mask].copy()


def print_stats(imdb):
    roidb = imdb['roidb']
    num_annos = 0
    num_crowd = 0
    num_dets = 0
    for roi in roidb:
        if 'gt_crowd' in roi:
            crowd = np.sum(roi['gt_crowd'])
            num_crowd += crowd
            num_annos += roi['gt_boxes'].shape[0] - crowd
        if 'dets' in roi:
            num_dets += roi['dets'].shape[0]
    print('{} images: {} detections, {} crowd annotations, {} non-crowd annotations'.format(
        len(roidb), num_dets, num_crowd, num_annos))


def get_avg_batch_size(imdb):
    num_dets = sum(roi['dets'].shape[0] for roi in imdb['roidb'] if 'dets' in roi)
    num_imgs = len(imdb['roidb'])
    return num_dets / num_imgs
