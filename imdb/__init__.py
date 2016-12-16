from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imdb.coco import load_coco

# imdb needs to contain:
#   image path
#   annotations
#   detections


# imdb_name -> function to generate it
_imdbs = {}

for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'minival2']:
        name = 'coco_{}_{}'.format(year, split)
        _imdbs[name] = lambda split=split, year=year: load_coco(split, year)

for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        _imdbs[name] = lambda split=split, year=year: load_coco(split, year)

def get_imdb(name):
    return _imdbs[name]()