from yacs.config import CfgNode as CN

# Miscellaneous configs
_C = CN()

# Experiment related
_C.exp = CN()
_C.exp.name           = ''
_C.exp.checkpoint_dir = ''
_C.exp.weight_file    = ''
_C.exp.gpus           = [0]
_C.exp.num_workers    = 8
_C.exp.batch_size     = 1
_C.exp.num_epochs     = 1000
_C.exp.log_dir        = ''
_C.exp.load_from      = ''

# Model related
_C.model = CN()
_C.model.encoder      = ''
_C.model.encoder_geo  = ''
_C.model.pose_predictor_quat = ''
_C.model.pose_predictor_rot = ''
_C.model.pose_predictor_trans = ''
_C.model.corr_module  = ''
_C.model.sdf_predictor= ''
_C.model.aggregator   = ''
_C.model.pc_feat_dim = 512 # 这里调feature的dimision
_C.model.transformer_feat_dim = 1024
_C.model.num_heads   = 4
_C.model.num_blocks  = 1
_C.model.recon_loss = False
_C.model.point_loss = False
_C.model.corr_feat = 'False'

# Data related
_C.data = CN()
_C.data.root_dir       = ''
_C.data.train_csv_file = ''
_C.data.val_csv_file  = ''
_C.data.num_pc_points  = 1024

# Optimizer related
_C.optimizer = CN()
_C.optimizer.lr           = 1e-3
_C.optimizer.lr_decay     = 0.7
_C.optimizer.decay_step   = 2e4
_C.optimizer.weight_decay = 1e-6
_C.optimizer.lr_clip      = 1e-5

def get_cfg_defaults():
    return _C.clone()