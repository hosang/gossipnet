from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

try:
    import cPickle as pickle
except ImportError:
    import pickle

from imdb.coco import load_coco
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


def get_imdb(name):
    cache_filename = os.path.join(
        cfg.ROOT_DIR, 'data', 'cache',
        '{}_{}_imdb_cache.pkl'.format(name, cfg.train.detector))
    if os.path.exists(cache_filename):
        print('reading {}'.format(cache_filename))
        with open(cache_filename, 'rb') as fp:
            return pickle.load(fp)

    result_imdb = _imdbs[name]()
    with open(cache_filename, 'wb') as fp:
        pickle.dump(result_imdb, fp)
    print('wrote {}'.format(cache_filename))
    return result_imdb


def prepro_train(train_imdb):
    imdb.tools.print_stats(train_imdb)
    if cfg.train.only_class != '':
        print('dropping all classes but {}'.format(cfg.train.only_class))
        imdb.tools.only_keep_class(train_imdb, cfg.train.only_class)
    print('dropping images without detections')
    train_imdb['roidb'] = imdb.tools.drop_no_dets(train_imdb['roidb'])
    print('dropping images without annotations')
    train_imdb['roidb'] = imdb.tools.drop_no_gt(train_imdb['roidb'])
    print('appending flipped images')
    train_imdb['roidb'] = imdb.tools.append_flipped(train_imdb['roidb'])
    train_imdb['avg_num_dets'] = imdb.tools.get_avg_batch_size(train_imdb)
    imdb.tools.print_stats(train_imdb)

