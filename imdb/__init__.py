from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

try:
    import cPickle as pickle
except ImportError:
    import pickle

from imdb.coco import load_coco
from imdb.pal import load_pal
import imdb.tools
from nms_net import cfg

# imdb needs to contain:
#   image path
#   annotations
#   detections


# imdb_name -> function to generate it
_imdbs = {}

for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'minival2',
                  'debug']:
        name = 'coco_{}_{}'.format(year, split)
        _imdbs[name] = lambda split=split, year=year: load_coco(split, year)

for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        _imdbs[name] = lambda split=split, year=year: load_coco(split, year)

# city persons
imsize = (2048, 1024)
for version in ['', '_synth', '_synth25k']:
    for split in ['train', 'val', 'test']:
        name = 'citypersons{}_{}'.format(version, split)
        has_gt = split == 'train'
        _imdbs[name] = lambda name=name, has_gt=has_gt, imsize=imsize: load_pal(
            name, has_gt, imsize=imsize)


def get_imdb(name, is_training):
    cache_filename = os.path.join(
        cfg.ROOT_DIR, 'data', 'cache',
        '{}_{}_imdb_cache.pkl'.format(name, cfg.train.detector))
    if os.path.exists(cache_filename):
        print('reading {}'.format(cache_filename))
        with open(cache_filename, 'rb') as fp:
            result_imdb = pickle.load(fp)
    else:
        result_imdb = _imdbs[name]()
        with open(cache_filename, 'wb') as fp:
            pickle.dump(result_imdb, fp)
        print('wrote {}'.format(cache_filename))

    if is_training:
        prepro_train(result_imdb)
    else:
        prepro_test(result_imdb)
    return result_imdb


def prepro_test(test_imdb):
    print('preparing test imdb')
    imdb.tools.print_stats(test_imdb)
    if cfg.train.only_class != '':
        print('dropping all classes but {}'.format(cfg.train.only_class))
        imdb.tools.only_keep_class(test_imdb, cfg.train.only_class)
        imdb.tools.print_stats(test_imdb)
    print('dropping images without detections')
    test_imdb['roidb'] = imdb.tools.drop_no_dets(test_imdb['roidb'])
    imdb.tools.print_stats(test_imdb)
    print('done')


def prepro_train(train_imdb):
    print('preparing train imdb')
    imdb.tools.print_stats(train_imdb)
    if cfg.train.only_class != '':
        print('dropping all classes but {}'.format(cfg.train.only_class))
        imdb.tools.only_keep_class(train_imdb, cfg.train.only_class)
        imdb.tools.print_stats(train_imdb)
    print('dropping images without detections')
    train_imdb['roidb'] = imdb.tools.drop_no_dets(train_imdb['roidb'])
    imdb.tools.print_stats(train_imdb)
    if cfg.train.max_num_detections > 0:
       print('dropping all but {} highest scoring detections'.format(
           cfg.train.max_num_detections))
       imdb.tools.drop_too_many_detections(train_imdb,
                                           cfg.train.max_num_detections)
       imdb.tools.print_stats(train_imdb)
    # print('dropping images without annotations')
    # train_imdb['roidb'] = imdb.tools.drop_no_gt(train_imdb['roidb'])
    print('appending flipped images')
    train_imdb['roidb'] = imdb.tools.append_flipped(train_imdb['roidb'])
    train_imdb['avg_num_dets'] = imdb.tools.get_avg_batch_size(train_imdb)
    imdb.tools.print_stats(train_imdb)
    print('done')


if __name__ == '__main__':
    from imdb import vis
    cfg.train.detector = 'FRCN_train'
    a_imdb = get_imdb('citypersons_train', True)
    vis.visualize_roidb(a_imdb['roidb'])

