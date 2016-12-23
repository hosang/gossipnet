
import yaml
from easydict import EasyDict as edict
import numpy as np
import os.path


this_path = os.path.dirname(os.path.realpath(__file__))

cfg = edict()
cfg.random_seed = 42
cfg.prefetch_q_size = 20
cfg.log_dir = './log'
cfg.ROOT_DIR = os.path.normpath(os.path.join(this_path, '..'))
cfg.resnet_type = '50'
cfg.imfeat_crop_width = 7
cfg.imfeat_crop_height = 7
cfg.pixel_mean = [123.68, 116.779, 103.939]

# resize the image, so the shorter side becomes `image_target_size`
cfg.image_target_size = 600
# resize the image, so the longer side is not longer than `image_max_size`
cfg.image_max_size = 1000


# training parameters
cfg.train = edict()
cfg.train.optimizer = 'adam'
cfg.train.model_init = None
cfg.train.resume = None
cfg.train.momentum = 0.9
cfg.train.weight_decay = 0.0005
cfg.train.num_iter = 100000
cfg.train.save_iter = 10000
cfg.train.lr_multi_step = [(10000, 0.001), (80000, 0.0001), (200000, 0.0000001)]
cfg.train.gradient_clipping = -1.0  # 1000.0
cfg.train.detector = 'FRCN_person'
cfg.train.flip = True
cfg.train.only_class = ''
cfg.train.imdb = 'coco_2014_train'
cfg.train.pos_weight = 0.1
cfg.train.pretrained_model = ''
cfg.train.display_iter = 20
cfg.train.det_min_size = 4

# test parameters
cfg.test = edict()
cfg.test.imdb = 'coco_2014_minival'

# Gnet parameters
cfg.gnet = edict()
cfg.gnet.neighbor_thresh = 0.2
cfg.gnet.shortcut_dim = 128
cfg.gnet.num_blocks = 16
cfg.gnet.reduced_dim = 32
cfg.gnet.pairfeat_dim = 2 * 32
cfg.gnet.gt_match_thresh = 0.5
cfg.gnet.num_block_pw_fc = 2
cfg.gnet.num_block_fc = 2
cfg.gnet.num_predict_fc = 3
cfg.gnet.block_dim = 2 * 32
cfg.gnet.predict_fc_dim = 128
cfg.gnet.imfeats = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)

