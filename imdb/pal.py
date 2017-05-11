
import os
import pickle
import itertools
import collections

from scipy.misc import imread
import numpy as np

from nms_net import cfg
from imdb.file_formats import pal
from imdb.tools import validate_boxes
from imdb.coco import merge_roidbs


def load_pal(dataset_name, has_gt, min_height=50, min_vis=0.65, imsize=None):
    id_generator = collections.defaultdict(itertools.count().__next__)
    gt_file = os.path.join(cfg.ROOT_DIR, 'data',
                           '{}.pal'.format(dataset_name))
    det_file = os.path.join(cfg.ROOT_DIR, 'data',
                            '{}_{}.pal'.format(
                                dataset_name, cfg.train.detector))

    print('loading detections')
    roidb = _load_dets(det_file, id_generator)
    if has_gt:
        print('loading annotations')
        gt_roidb = _load_gt(gt_file, id_generator, min_height, min_vis)
        roidb = merge_roidbs(roidb, gt_roidb)

    print('loading image information')
    add_iminfo(roidb, imsize)

    classes = ('__background__', 'person')
    class_to_ind = {cl: ind for ind, cl in enumerate(classes)}
    class_to_cat_id = {classes[1]: 1}

    imdb = {
        'name': dataset_name,
        'classes': classes,
        'class_to_ind': class_to_ind,
        'class_to_cat_id': class_to_cat_id,
        'num_classes': len(classes),
        'roidb': roidb,
    }
    return imdb


def _load_dets(det_file, id_generator):
    min_size = cfg.train.det_min_size
    roidb = pal.load(det_file, id_generator, min_height=min_size, min_vis=-1)
    for roi in roidb:
        valid = np.logical_not(roi['ignore'])
        del roi['ignore']
        roi['dets'] = roi['boxes'][valid, :]
        del roi['boxes']
        roi['det_scores'] = roi['scores'][valid]
        del roi['scores']
        roi['det_classes'] = roi['classes'][valid]
        del roi['classes']
    return roidb


def _load_gt(gt_file, id_generator, min_height, min_vis):
    roidb = pal.load(gt_file, id_generator, min_height=min_height,
                     min_vis=min_vis)
    for roi in roidb:
        valid = np.logical_not(roi['ignore'])
        del roi['scores']
        roi['gt_boxes'] = roi['boxes'][valid, :]
        del roi['boxes']
        roi['gt_crowd'] = roi['ignore'][valid]
        del roi['ignore']
        roi['gt_classes'] = roi['classes'][valid]
        del roi['classes']
    return roidb


def add_iminfo(roidb, imsize):
    if imsize is not None:
        w, h = imsize

    for roi in roidb:
        if imsize is None:
            im = imread(roi['filename'])
            h, w = im.shape[:2]
        roi['width'] = w
        roi['height'] = h
        roi['flipped'] = False

        crop_to_im_boundaries(roi['dets'], w, h)
        crop_to_im_boundaries(roi['gt_boxes'], w, h)
        validate_boxes(roi['dets'], width=w, height=h)
        validate_boxes(roi['gt_boxes'], width=w, height=h)


def crop_to_im_boundaries(boxes, width, height):
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width)
    boxes[:, 3] = np.minimum(boxes[:, 3], height)