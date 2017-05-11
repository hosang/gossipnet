#!/home/jhosang/env/py3.4-tensorflow/bin/python3.4

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.misc
import os.path
import matplotlib.colors

sys.path.insert(1, '/BS/bbox_nms_net2/work/src/coco/PythonAPI')
sys.path.insert(1, '/BS/bbox_nms_net2/work/src/coco_multiclass_nms')

import pycocotools.mask as mask_utils



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
    fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax, aspect='normal')
    # plt.imshow(im[:, :, (2, 1, 0)])
    plt.imshow(im)


def add_noise(dets, seed=42):
    dets = dets.copy()
    np.random.seed(seed)
    n = dets.shape[0]
    sample = np.random.rand(n) > 0.2
    shape = dets[sample, 4].shape
    new_score = dets[sample, 4] + np.random.rand(*shape) * 0.7
    new_score = np.minimum(np.amax(dets[:, 4]), new_score)
    dets[sample, 4] = new_score
    return dets


def plot_proposals(dets, crop, score_weight=False):
    # if isinstance(dets, list):
    #     # print(dets[0])
    #     list_dets = dets
    #     n = len(dets)
    #     dets = np.zeros((n, 5), dtype=np.float32)
    #     for i, dt in enumerate(list_dets):
    #         assert isinstance(dt, dict)
    #         if 'bbox' not in dt:
    #             dt['bbox'] = mask_utils.toBbox(dt['segmentation'])
    #         x1, y1, width, height = dt['bbox']
    #         score = dt.get('score', 1.0)
    #         dets[i, :] = (x1, y1, x1 + width, y1 + height, score)
    #     dets[:, 4] = 1.0 / (1.0 + np.exp(-dets[:, 4]))

    for i in range(dets.shape[0]):
        x1, y1, x2, y2, score = dets[i, :]
        width = x2 - x1
        height = y2 - y1
        x1 -= crop[0]
        y1 -= crop[1]
        if score_weight:
            alpha = score
        else:
            alpha = 0.7
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), width, height,
            fill=False, edgecolor='#e41a1c', alpha=alpha, linewidth=linewidth,
            linestyle='-',
        ))


def plot_matching_labels(dets, is_matched, crop, score_weight=False):
    tp_color = '#377eb8'
    fp_color = '#ff7f00'
    tp_color = '#377eb8'
    fp_color = '#e41a1c'

    tp_color = '#6cd140'
    fp_color = '#ffff33'

    n = dets.shape[0]
    for i in range(n - 1, -1, -1):
        color = tp_color if is_matched[i] > 0 else fp_color

        x1, y1, x2, y2, score = dets[i, :]
        width = x2 - x1
        height = y2 - y1
        x1 -= crop[0]
        y1 -= crop[1]
        if score_weight:
            alpha = score
        else:
            alpha = 1.0
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), width, height,
            fill=False, edgecolor=color, alpha=alpha, linewidth=linewidth,
            linestyle='-',
        ))


def pick_eval_dets(dets, eval_im):
    dts = {d['id']: d for d in dets}
    t = 0
    n = len(eval_im['dtIds'])
    out_dets = np.zeros((n, 5), dtype=np.float32)
    det_matched = np.zeros((n,), dtype=np.int8)
    for i in range(n):
        dt = dts[eval_im['dtIds'][i]]
        x1, y1, width, height = dt['bbox']
        x2 = x1 + width
        y2 = y1 + height
        score = dt['score']
        out_dets[i, :] = (x1, y1, x2, y2, score)

        is_ignored = eval_im['dtIgnore'][t, i]
        # assert not is_ignored
        is_matched = eval_im['dtMatches'][t, i] > 0
        if is_matched:
            det_matched[i] = 1
    # out_dets[:, 4] = 1.0 / (1.0 + np.exp(-out_dets[:, 4]))
    return out_dets, det_matched


def evaldets_to_dets(dets):
    n = len(dets)
    out_dets = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        dt = dets[i]
        x1, y1, width, height = dt['bbox']
        x2 = x1 + width
        y2 = y1 + height
        score = dt['score'] if 'score' in dt else 1.0
        out_dets[i, :] = (x1, y1, x2, y2, score)
    # out_dets[:, 4] = 1.0 / (1.0 + np.exp(-out_dets[:, 4]))
    return out_dets


def dets_to_scoremap(dets, gridwidth, crop):
    cropwidth = crop[2] - crop[0]
    cropheight = crop[3] - crop[1]
    gridheight = int(round(gridwidth / cropwidth * cropheight))
    xy = (dets[:, 2:4] + dets[:, 0:2]) / 2.0
    map = np.zeros((gridheight, gridwidth), dtype=np.float32)
    for i in range(dets.shape[0]):
        x = (xy[i, 0] - crop[0]) / cropwidth * gridwidth
        y = (xy[i, 1] - crop[1]) / cropheight * gridheight
        x = int(np.floor(x))
        y = int(np.floor(y))
        if 0 <= x < gridwidth and 0 <= y < gridheight:
            map[y, x] = max(map[y, x], dets[i, 4])
            # if dets[i, 4] > map[y, x]:
    return map


def show_as_grid(dets, crop, gridwidth=20, noise=0.0):
    if noise > 0:
        np.random.seed(42)
        dets = dets.copy()
        pos = dets[:, :2]
        pos += np.random.normal(scale=noise, size=pos.shape) * (1.0 / dets[:, 4][:, None])
    map = dets_to_scoremap(dets, gridwidth, crop)

    aspect = float(map.shape[1]) / map.shape[0]
    fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax, aspect='normal')
    plt.imshow(map, interpolation='nearest', cmap='magma')


def main():
    eval_file = '/BS/bbox_nms_net2/work/exp-tf/2017-03-02_person_depth_sweep_embedded_pwfeats/2017-03-02_single_class_person_8block_lowerlr_seed47_embedded/gnet-1920000_coco_2014_valminusminival/detections.scores.results.pkl'
    detection_file = '/BS/bbox_nms_net2/work/coco_FRCN_detections/valminusminival_pre_detections_nms.pkl'
    nms_detection_file = '/BS/bbox_nms_net2/work/coco_FRCN_detections/valminusminival_detections_nms0.5.pkl'
    image_filename = '/BS/databases/coco/images/val2014/COCO_val2014_000000000192.jpg'
    crop = (260, 168, 534, 480)
    # crop = (0, 0, 534, 480)
    cache_file = 'det_cache.pkl'

    if os.path.exists(cache_file):
        print('loading cache')
        d = pickle.load(open(cache_file, 'rb'), encoding='latin1')
    else:
        images, evalImgs, gts, dts = load_eval_file(eval_file, 1)
        im_to_id = {fn: imid for imid, fn in images.items()}
        imid = im_to_id[os.path.basename(image_filename)]
        print(imid)
        detector_dets = load_dets(detection_file, 1, imid)
        detector_dets_nms = load_dets(nms_detection_file, 1, imid)

        im = scipy.misc.imread(image_filename)
        im = im[crop[1]:crop[3], crop[0]:crop[2], :].copy()

        gt = evaldets_to_dets(gts[imid])
        gnet_all_dets = evaldets_to_dets(dts[imid])
        gnet_dets, gnet_matching = pick_eval_dets(dts[imid], evalImgs[imid])
        d = {
            'im': im,
            'gt': gt,
            'frcn_dets': detector_dets,
            'frcn_dets_nms': detector_dets_nms,
            'gnet_dets': gnet_dets,
            'gnet_dets_all': gnet_all_dets,
            'gnet_matching': gnet_matching,
        }

        print('dumping cache')
        with open(cache_file, 'wb') as fp:
            pickle.dump(d, fp)

    for k in ('gnet_dets', 'gnet_dets_all'):
        d[k][:, 4] = 1.0 / (1.0 + np.exp(-d[k][:, 4]))
    for k in ('frcn_dets', 'frcn_dets_nms', 'gnet_dets', 'gnet_dets_all'):
        print(k, np.min(d[k][:, 4]), np.max(d[k][:, 4]))

    print(d['im'].min(), d['im'].max())
    dim_im = matplotlib.colors.rgb_to_hsv(d['im'] / 255.0)
    print(dim_im.min(), dim_im.max())
    dim_im[:, :, 1] *= 0.5
    dim_im = matplotlib.colors.hsv_to_rgb(dim_im)

    gridwidth = 30

    print('image')
    plot_image(d['im'])
    plt.savefig('00_image.png')

    print('gt')
    plot_image(dim_im)
    plot_proposals(d['gt'], crop)
    plt.savefig('01_gt.png')

    show_as_grid(d['gt'], crop, gridwidth=gridwidth)
    plt.savefig('02_gt_scoremap.png')
    show_as_grid(d['gt'], crop, gridwidth=40)
    plt.savefig('03_gt_scoremap.png')

    print('proposals')
    plot_image(dim_im)
    plot_proposals(d['frcn_dets'], crop)
    plt.savefig('10_proposals_frcn.png')

    plot_image(dim_im)
    plot_proposals(d['gnet_dets'], crop)
    plt.savefig('11_proposals_gnet_eval.png')

    plot_image(dim_im)
    plot_proposals(d['gnet_dets_all'], crop)
    plt.savefig('12_proposals_gnet_all.png')

    print('scored proposals')
    plot_image(dim_im)
    plot_proposals(d['frcn_dets'], crop, score_weight=True)
    plt.savefig('20_scored_proposals.png')

    show_as_grid(d['frcn_dets'], crop, gridwidth=gridwidth)
    plt.savefig('21_scored_proposals_scoremap.png')
    show_as_grid(d['frcn_dets'], crop, gridwidth=40, noise=8)
    plt.savefig('22_scored_proposals_scoremap.png')

    print('nmsed scored proposals')
    plot_image(dim_im)
    plot_proposals(d['frcn_dets_nms'], crop, score_weight=True)
    plt.savefig('30_nms_scored_proposals.png')

    print('nmsed scored proposals')
    plot_image(dim_im)
    plot_proposals(d['frcn_dets_nms'], crop, score_weight=False)
    plt.savefig('31_nms_proposals.png')

    show_as_grid(d['frcn_dets_nms'], crop, gridwidth=gridwidth)
    plt.savefig('32_nms_scored_proposals_scoremap.png')

    print('gnet dets')
    plot_image(dim_im)
    plot_proposals(d['gnet_dets'], crop, score_weight=True)
    plt.savefig('40_gnet_rescored_dets.png')

    gnet_dets_noisy = add_noise(d['gnet_dets'])
    plot_image(dim_im)
    plot_proposals(gnet_dets_noisy, crop, score_weight=True)
    plt.savefig('41_gnet_rescored_dets_noise.png')

    plot_image(dim_im)
    plot_proposals(d['gnet_dets_all'], crop, score_weight=True)
    plt.savefig('42_gnet_rescored_dets_all.png')

    plot_image(dim_im)
    plot_matching_labels(d['gnet_dets'], d['gnet_matching'], crop, score_weight=True)
    plt.savefig('50_matching_labels_weighted.png')

    plot_image(dim_im)
    plot_matching_labels(d['gnet_dets'], d['gnet_matching'], crop, score_weight=False)
    plt.savefig('51_matching_labels.png')

    plot_image(dim_im)
    plot_matching_labels(gnet_dets_noisy, d['gnet_matching'], crop, score_weight=True)
    plt.savefig('52_matching_labels_weighted_noise.png')

if __name__ == '__main__':
    main()
