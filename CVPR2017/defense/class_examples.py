#!/home/jhosang/env/py3.4-tensorflow/bin/python3.4

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.misc
import os.path
import matplotlib.colors
import argparse
import errno

sys.path.insert(1, '/BS/bbox_nms_net2/work/src/coco/PythonAPI')
sys.path.insert(1, '/BS/bbox_nms_net2/work/src/coco_multiclass_nms')

import pycocotools.mask as mask_utils
from pycocotools.coco import COCO


linewidth = 4.0

def filter_class(box_set, want_catid):
    # can be used with gts and with dts
    res = {}
    for (im_id, cat_id), gt in box_set.items():
        if cat_id != want_catid:
            continue
        res[im_id] = gt
    return res


def load_eval_file(eval_result_file, catid):
    print('reading {}'.format(eval_result_file))
    eval = pickle.load(open(eval_result_file, 'rb'), encoding='latin1')
    print('preparing...')
    assert 'dts' in eval, 'need detections saved in the eval file'
    assert 'evalImgs' in eval, 'need evalImgs saved in the eval file'

    images = {i['id']: i['file_name'] for i in eval['images']}
    evalImgs = {e['image_id']: e
                for e in eval['evalImgs']
                if e is not None and e['category_id'] == catid}
    gts = filter_class(eval['gts'], catid)
    dts = filter_class(eval['dts'], catid)
    return images, evalImgs, gts, dts


def load_det_pkl_file(det_filename):
    print('reading {}'.format(det_filename))
    with open(det_filename, 'rb') as fp:
        all_boxes = pickle.load(fp, encoding='latin1')
        image_ids = class_ids = None
        if isinstance(all_boxes, tuple) and len(all_boxes) == 3:
            all_boxes, image_ids, class_ids = all_boxes
    print('done')

    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    if image_ids is None:
        image_ids = [-1] * num_images
        class_ids = [-1] * num_classes
    return all_boxes, image_ids, class_ids
    # for im_idx in range(num_images):
    #     dets = [
    #         (all_boxes[cls][im_idx]
    #          if not isinstance(all_boxes[cls][im_idx], list)
    #          else np.zeros((0, 5), dtype=np.float32))
    #         for cls in range(num_classes)]
    #     yield (image_ids[im_idx], class_ids, dets)


def load_dets(filename, cat_id, imid):
    dets, imids, class_ids = load_det_pkl_file(filename)
    im_idx = imids.index(imid)
    cat_idx = class_ids.index(cat_id)
    imdets = dets[cat_idx][im_idx]
    if isinstance(imdets, list):
        imdets = np.zeros((0, 5), dtype=np.float32)
    return imdets


def plot_image(im):
    aspect = float(im.shape[1]) / im.shape[0]
    fig = plt.figure(frameon=False, figsize=(6 * aspect, 6))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax, aspect='normal')
    # plt.imshow(im[:, :, (2, 1, 0)])
    plt.imshow(im)


def plot_annotations(dets, crop=(0, 0, 0, 0)):
    alpha = 0.8

    for det in dets:
        x1, y1, width, height = det['bbox']
        x1 -= crop[0]
        y1 -= crop[1]
        ls = '--' if det['iscrowd'] else '-'
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), width, height,
            fill=False, edgecolor='#e41a1c', alpha=alpha, linewidth=linewidth,
            linestyle=ls,
        ))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', default='bed')
    args = parser.parse_args()

    ann_file = '/BS/bbox_nms_net2/work/coco_annotations/instances_val2014.json'
    img_dir = '/BS/databases/coco/images/val2014'
    outpath_mask = 'class_examples/{class_id}_{class_name}'
    browse = True

    class_names = ['bed', 'couch', 'surfboard', 'cat', 'dog',
                   'apple', 'mouse', 'bear', 'fire hydrant', 'giraffe']
    class_names = ['person']
    num = 200

    coco = COCO(ann_file)
    for class_name in class_names:
        catIds = coco.getCatIds(catNms=[class_name])
        imgIds = coco.getImgIds(catIds=catIds)
        print('{} {} {}'.format(class_name, len(imgIds),
                             len(coco.getAnnIds(catIds=catIds, iscrowd=None))))
        # continue
        for idx, imid in enumerate(imgIds[:num]):
            img = coco.loadImgs(imid)[0]
            image_filename = os.path.join(img_dir, img['file_name'])
            im = scipy.misc.imread(image_filename)
            dim_im = matplotlib.colors.rgb_to_hsv(im / 255.0)
            dim_im[:, :, 1] *= 0.5
            # dim_im[:, :, 2] = 0.5 + 0.5 * dim_im[:, :, 2]
            dim_im = matplotlib.colors.hsv_to_rgb(dim_im)

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            print(img['id'])

            plot_image(dim_im)
            plot_annotations(anns)

            outpath = outpath_mask.format(class_name=class_name, class_id=catIds[0])
            mkdir_p(outpath)
            out_file = os.path.join(outpath, '{:05d}.png'.format(idx))
            plt.savefig(out_file)
            plt.close()


if __name__ == '__main__':
    main()
