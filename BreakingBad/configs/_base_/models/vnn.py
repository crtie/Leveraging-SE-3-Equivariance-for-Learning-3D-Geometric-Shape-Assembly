"""VNN model."""
from yacs.config import CfgNode as CN
_C = CN()
_C.name = 'vnn'
_C.rot_type = 'rmat'

_C.encoder = 'vn_dgcnn'
_C.regressor = 'original'
_C.point_loss = 'point'
_C.translation_embedding = False
_C.check_equiv = True
_C.with_corr = True
_C.with_adv = False
_C.rot_loss = 'geo'  #! 'geo' or 'L1'
_C.pointloss = 'L2' #! 'L2' or 'cham'
_C.pc_feat_dim = 512
_C.close_eps = 0.05
_C.GoverD = 3



def get_cfg_defaults():
    return _C.clone()