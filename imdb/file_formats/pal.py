
from collections import namedtuple
import numpy as np

from . import AnnoList_pb2


Annotation = namedtuple('Annotation', ['rect', 'ignore', 'score'])


def _load_proto(filename):
    _annolist = AnnoList_pb2.AnnoList()

    f = open(filename, "rb")
    _annolist.ParseFromString(f.read())
    f.close()

    return _annolist


def load(filename, min_height, min_vis, use_order_as_ids=True,
         id_generator=None):
    _annolist = _load_proto(filename)
    annotations = []
    seen_imids = set()

    num_boxes = num_too_small = num_too_occluded = num_ignore = 0

    for idx, _a in enumerate(_annolist.annotation):
        image_filename = _a.imageName
        if use_order_as_ids:
            im_id = idx
        else:
            im_id = id_generator[image_filename]
        annos = []

        for _r in _a.rect:
            rect = (_r.x1, _r.y1, _r.x2, _r.y2)
            assert rect[2] > rect[0]
            assert rect[3] > rect[1]
            box_height = _r.y2 - _r.y1

            ignore = False
            if _r.HasField("track_id"):
                ignore = 0 if _r.track_id == 1 else 1
                if ignore:
                    num_ignore += 1

            score = -1.0
            if _r.HasField("score"):
                score = _r.score

            points = []
            for _p in _r.point:
                points.append((_p.x, _p.y))

            if len(points) >= 3:
                vis_x1, vis_y1 = points[0]
                vis_x2, vis_y2 = points[2]

                vis_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
                total_area = (_r.x2 - _r.x1) * (_r.y2 - _r.y1)
                vis_ratio = vis_area / float(total_area)

                if not ignore and vis_ratio < min_vis:
                    ignore = 1
                    num_too_occluded += 1
            if not ignore and box_height < min_height:
                ignore = 1
                num_too_small += 1

            anno = Annotation(rect=rect, ignore=ignore, score=score)
            annos.append(anno)

        n = len(annos)
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.zeros((n,), dtype=np.float32)
        classes = np.ones((n,), dtype=np.int32)
        ignore = np.zeros((n,), dtype=np.uint8)
        for i, anno in enumerate(annos):
            boxes[i, :] = anno.rect
            scores[i] = anno.score
            ignore[i] = anno.ignore
        num_boxes += n

        assert im_id not in seen_imids
        seen_imids.add(im_id)
        annotations.append({
            'id': im_id,
            'filename': image_filename,
            'boxes': boxes,
            'ignore': ignore,
            'scores': scores,
            'classes': classes,
        })

    print('{} images with {} boxes; of those: {} ignore, {} too small, '
          '{} too occluded'.format(
              len(annotations), num_boxes, num_ignore, num_too_small,
              num_too_occluded))
    return annotations
