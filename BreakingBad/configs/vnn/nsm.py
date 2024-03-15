import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../_base_/default_exp.py',
    'data': '../_base_/datasets/breaking_bad/everyday.py',
    'optimizer': '../_base_/schedules/adam_cosine.py',
    'model': '../_base_/models/vnn.py',
    'loss': '../_base_/models/loss/vnn_loss.py',
}

# Miscellaneous configs
_C = CN()


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    cfg.model.pointloss = 'cham'
    cfg.model.encoder = 'dgcnn'
    cfg.model.pc_feat_dim = 256
    cfg.loss.trans_loss_w = 50.
    cfg.loss.recons_loss_w = 5.
    cfg.loss.point_loss_w = 100.
    
    return cfg
