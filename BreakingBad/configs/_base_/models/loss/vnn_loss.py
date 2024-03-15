from yacs.config import CfgNode as CN

_C = CN()
_C.noise_dim = 0  # no stochastic

_C.trans_loss_w = 1.
_C.rot_loss_w = 1.
_C.recons_loss_w = 1.
_C.point_loss_w = 1.
_C.embedding_loss_w = 1.
_C.adv_loss_w = 0.1
_C.adv_G_loss_w = 0.1
_C.adv_D_loss_w = 0.1
_C.noise_dim = 0  # no stochastic
def get_cfg_defaults():
    return _C.clone()
