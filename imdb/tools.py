from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

    roidb = imdb['roidb']
    for roi in roidb:
        mask = roi['gt_classes'] == cls_ind
        roi['gt_classes'] = roi['gt_classes'][mask].copy()
        roi['gt_boxes'] = roi['gt_boxes'][mask].copy()
        roi['gt_crowd'] = roi['gt_crowd'][mask].copy()

        mask = roi['det_classes'] == cls_ind
        roi['det_classes'] = roi['det_classes'][mask].copy()
        roi['dets'] = roi['dets'][mask].copy()
        roi['det_scores'] = roi['det_scores'][mask].copy()
