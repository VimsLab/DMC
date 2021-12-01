import os
import os.path as osp
import yaml


def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.makedirs(dir)
        else:
            raise Exception('Directory not exist: {}'.format(dir))


def get_train_config(config_file='config/train_config.yaml', dataset=""):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    _check_dir(cfg[dataset]['raw'], make_dir=False)
    _check_dir(cfg[dataset]['processed'])
    _check_dir(cfg[dataset]['ckpt_root'])

    return cfg


def get_test_config(config_file='config/test_config.yaml',  dataset=""):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    _check_dir(cfg[dataset]['processed'], make_dir=False)

    return cfg
