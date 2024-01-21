import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix
from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))
from models.encoder.pointnet import PointNet
from models.encoder.dgcnn import DGCNN
from models.decoder.MLPDecoder import MLPDecoder
from models.encoder.vn_dgcnn import VN_DGCNN, VN_DGCNN_corr, VN_DGCNN_New
from models.baseline.regressor_CR import Regressor_CR, Regressor_6d, VN_Regressor_6d
import utils
from pdb import set_trace
from ..ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


def debug_vis(batch_data, cfg, pred_data, iter_counts):
    save_dir = cfg.exp.log_dir
    vis_dir = os.path.join(save_dir, 'vis_res')
    if not os.path.exists((vis_dir)):
        os.mkdir(vis_dir)
    src_pc = batch_data['src_pc'][0]  # (3, 1024)
    tgt_pc = batch_data['tgt_pc'][0]  # (3, 1024)
    src_trans = pred_data['src_trans'][0].unsqueeze(1)  # (3)
    src_rot = pred_data['src_rot'][0]  # (6)
    tgt_trans = pred_data['tgt_trans'][0].unsqueeze(1)  # (3)
    tgt_rot = pred_data['tgt_rot'][0]  # (6)

    device = src_pc.device
    num_points = src_pc.shape[1]

    # src_rot_mat = quaternion_to_matrix(src_quat)  # (3, 3)
    # tgt_rot_mat = quaternion_to_matrix(tgt_quat)  # (3, 3)
    src_rot_mat = bgs(src_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
    tgt_rot_mat = bgs(tgt_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
    src_rot_mat = torch.linalg.inv(src_rot_mat)
    tgt_rot_mat = torch.linalg.inv(tgt_rot_mat)

    src_trans = src_trans.expand(-1, num_points)  # (3, 1024)
    tgt_trans = tgt_trans.expand(-1, num_points)  # (3, 1024)

    src_pc = torch.matmul(src_rot_mat.double(), src_pc.double())  # (3, 1024)
    tgt_pc = torch.matmul(tgt_rot_mat.double(), tgt_pc.double())  # (3, 1024)

    src_pc = src_pc + src_trans
    tgt_pc = tgt_pc + tgt_trans

    total_pc = np.array(torch.cat([src_pc, tgt_pc], dim=1).detach().cpu())
    color_mask_src = ['r'] * num_points
    color_mask_tgt = ['g'] * num_points
    total_color = color_mask_src + color_mask_tgt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(list(total_pc[0]), list(total_pc[1]), list(total_pc[2]), c=total_color, s=10, alpha=1)
    ax.axis('scaled')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    fig.savefig(os.path.join(vis_dir, '{}_vis_res{}'.format(batch_data['category_name'][0], iter_counts)))
    plt.close(fig)


def debug_vis_gt(batch_data, cfg, pred_data, iter_counts):
    save_dir = cfg.exp.log_dir
    vis_dir = os.path.join(save_dir, 'vis_res_gt')
    if not os.path.exists((vis_dir)):
        os.mkdir(vis_dir)
    src_pc = batch_data['src_pc'][0]  # (3, 1024)
    tgt_pc = batch_data['tgt_pc'][0]  # (3, 1024)
    src_trans = batch_data['src_trans'][0]  # (3, 1)
    src_rot = batch_data['src_rot'][0]  # (4)
    tgt_trans = batch_data['tgt_trans'][0]  # (3, 1)
    tgt_rot = batch_data['tgt_rot'][0]  # (4)
    device = src_pc.device
    num_points = src_pc.shape[1]

    src_rot_mat = bgs(src_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
    tgt_rot_mat = bgs(tgt_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
    src_rot_mat = torch.linalg.inv(src_rot_mat)
    tgt_rot_mat = torch.linalg.inv(tgt_rot_mat)

    # src_rot_mat = quaternion_to_matrix(src_quat)  # (3, 3)
    # tgt_rot_mat = quaternion_to_matrix(tgt_quat)  # (3, 3)

    src_trans = src_trans.expand(-1, num_points)  # (3, 1024)
    tgt_trans = tgt_trans.expand(-1, num_points)  # (3, 1024)

    src_pc = torch.matmul(src_rot_mat, src_pc)  # (3, 1024)
    tgt_pc = torch.matmul(tgt_rot_mat, tgt_pc)  # (3, 1024)

    src_pc = src_pc + src_trans
    tgt_pc = tgt_pc + tgt_trans

    total_pc = np.array(torch.cat([src_pc, tgt_pc], dim=1).detach().cpu())
    color_mask_src = ['r'] * num_points
    color_mask_tgt = ['g'] * num_points
    total_color = color_mask_src + color_mask_tgt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(list(total_pc[0]), list(total_pc[1]), list(total_pc[2]), c=total_color, s=10, alpha=1)
    ax.axis('scaled')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    fig.savefig(os.path.join(vis_dir, '{}_vis_res{}'.format(batch_data['category_name'][0], iter_counts)))
    plt.close(fig)


class ShapeAssemblyNet_vnn(nn.Module):

    def __init__(self, cfg, data_features):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder()
        self.pose_predictor_rot = self.init_pose_predictor_rot()
        self.pose_predictor_trans = self.init_pose_predictor_trans()
        if self.cfg.model.recon_loss:
            self.decoder = self.init_decoder()
        self.data_features = data_features
        self.iter_counts = 0
        self.close_eps = 0.1
        self.L2 = nn.MSELoss() 
        self.R = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
                  [0.53452248, -0.57735027, -0.6172134],
                  [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64).unsqueeze(0)
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()


    def init_encoder(self):
        if self.cfg.model.encoder == 'dgcnn':
            encoder = DGCNN(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == 'vn_dgcnn':
            encoder = VN_DGCNN_New(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == 'pointnet':
            encoder = PointNet(feat_dim=self.cfg.model.pc_feat_dim)
        return encoder

    def init_pose_predictor_rot(self):
        if self.cfg.model.encoder == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_rot == 'original':
            pose_predictor_rot = Regressor_CR(pc_feat_dim= pc_feat_dim, out_dim=6)
        elif self.cfg.model.pose_predictor_rot == 'vn':
            pose_predictor_rot = VN_equ_Regressor(pc_feat_dim= pc_feat_dim/3, out_dim=6)

        return pose_predictor_rot

    def init_pose_predictor_trans(self):
        if self.cfg.model.encoder == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_trans == 'original':
            pose_predictor_trans = Regressor_CR(pc_feat_dim=pc_feat_dim, out_dim=3)
        elif self.cfg.model.pose_predictor_trans == 'vn':
            pose_predictor_trans = VN_inv_Regressor(pc_feat_dim=pc_feat_dim/3, out_dim=3)
        return pose_predictor_trans

    def init_decoder(self):
        pc_feat_dim = self.cfg.model.pc_feat_dim
        decoder = MLPDecoder(feat_dim=pc_feat_dim, num_points=self.cfg.data.num_pc_points)
        return decoder


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer


    def check_equiv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(torch.matmul(x, R) - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return
    
    def check_inv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(x - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return

    def check_network_property(self, gt_data, pred_data):
        with torch.no_grad():
            B, _, N = gt_data['src_pc'].shape
            R = self.R.float().repeat(B, 1, 1).to(gt_data['src_pc'].device)
            pcs_R = torch.matmul(gt_data['src_pc'].permute(0, 2, 1), R).permute(0, 2, 1)
            pred_data_R = self.forward(pcs_R, gt_data['tgt_pc'])
            
            #! check pointcloud equivariance (should be naturally equivariant)
            # self.check_equiv(gt_data['src_pc'].reshape(B, N, 3), R.permute(0, 2, 1), pcs_R.reshape(B, N, 3), 'pcs')

            #! check equivariance of equivariant features
            equiv_feats = pred_data['Fa']
            equiv_feats_R = pred_data_R['Fa']
            self.check_equiv(equiv_feats, R, equiv_feats_R, 'equiv_feats')

            #! check invariance of invariant features
            inv_feats = pred_data['Ga']
            inv_feats_R = pred_data_R['Ga']
            self.check_inv(inv_feats, R, inv_feats_R, 'inv_feats')

            # if self.cfg.model.pose_predictor_trans == 'vn':
            #     self.check_inv(pred_data['src_trans'], R, pred_data_R['src_trans'], 'trans')

            if self.cfg.model.pose_predictor_rot == 'vn':
                rot = bgs(pred_data['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                rot_R = bgs(pred_data_R['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                # rot = pred_data['src_rot'].reshape(-1, 2, 3)
                # rot_R = pred_data_R['src_rot'].reshape(-1, 2, 3)
                self.check_equiv(rot, R, rot_R, 'rot')
        return

    def _recon_pts(self, Ga, Gb):
        global_inv_feat = torch.sum(torch.cat([Ga, Gb], dim=1), dim=1)
        recon_pts = self.decoder(global_inv_feat)
        return recon_pts

    def forward(self, src_pc, tgt_pc):
        batch_size = src_pc.shape[0]
        num_points = src_pc.shape[2]
        # print(src_pc.shape)
        # print(tgt_pc.shape)
        if self.cfg.model.encoder == 'dgcnn':
            src_point_feat = self.encoder(src_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))
            tgt_point_feat = self.encoder(tgt_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))

            src_feat = torch.mean(src_point_feat, dim=2)
            tgt_feat = torch.mean(tgt_point_feat, dim=2)

        if self.cfg.model.encoder == 'vn_dgcnn':
            Fa, Ga = self.encoder(src_pc)
            Fb, Gb = self.encoder(tgt_pc)
            # set_trace()
            if self.cfg.model.corr_module == 'yes':
                # src_feat = torch.bmm(Gb, Fa)
                # tgt_feat = torch.bmm(Ga, Fb)
                src_feat = Fa * Gb[:, :, :3]
                tgt_feat = Fb * Ga[:, :, :3]
            else:
                src_feat = Fa
                tgt_feat = Fb
        if self.cfg.model.pose_predictor_rot == 'original':
            src_rot = self.pose_predictor_rot(src_feat.reshape(batch_size, -1))
            tgt_rot = self.pose_predictor_rot(tgt_feat.reshape(batch_size, -1))
        else:
            src_rot = self.pose_predictor_rot(src_feat)
            tgt_rot = self.pose_predictor_rot(tgt_feat)
        
        if self.cfg.model.pose_predictor_trans == 'original':
            src_trans = self.pose_predictor_trans(src_feat.reshape(batch_size, -1))
            tgt_trans = self.pose_predictor_trans(tgt_feat.reshape(batch_size, -1))
        else:
            src_trans = self.pose_predictor_trans(src_feat)
            tgt_trans = self.pose_predictor_trans(tgt_feat)
        
        if self.cfg.model.recon_loss:
            recon_pts = self._recon_pts(Ga, Gb)
        # 最后的返回值
        pred_dict = {
            'src_rot': src_rot,
            'src_trans': src_trans,
            'tgt_rot': tgt_rot,
            'tgt_trans': tgt_trans,
        }
        if self.cfg.model.encoder == 'vn_dgcnn':
            pred_dict['Fa'] = Fa
            pred_dict['Ga'] = Ga
        if self.cfg.model.recon_loss:
            pred_dict['recon_pts'] = recon_pts
        return pred_dict

    def compute_point_loss(self, batch_data, pred_data):
        # Point clouds

        src_pc = batch_data['src_pc'].float()  # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024

        # Ground truths
        src_rot_gt = self.recover_R_from_6d(batch_data['src_rot'].float())
        tgt_rot_gt = self.recover_R_from_6d(batch_data['tgt_rot'].float())
        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        # print('src_trans_gt: ', src_trans_gt.shape)

        # Model predictions
        src_rot_pred = self.recover_R_from_6d(pred_data['src_rot'].float())
        tgt_rot_pred = self.recover_R_from_6d(pred_data['tgt_rot'].float())
        src_trans_pred = pred_data['src_trans'].float()
        tgt_trans_pred = pred_data['tgt_trans'].float()

        src_trans_pred = src_trans_pred.unsqueeze(2)
        tgt_trans_pred = tgt_trans_pred.unsqueeze(2)

        # Source point loss
        transformed_src_pc_pred = src_rot_pred @ src_pc + src_trans_pred  # batch x 3 x 1024
        with torch.no_grad():
            transformed_src_pc_gt = src_rot_gt @ src_pc + src_trans_gt  # batch x 3 x 1024
        # transformed_src_pc_pred = self.batch_quat2mat(src_quat_pred) @ src_pc + src_trans_pred    # batch x 3 x 1024
        # transformed_src_pc_gt = self.batch_quat2mat(src_quat_gt) @ src_pc + src_trans_gt
        src_point_loss = torch.mean(torch.sum((transformed_src_pc_pred - transformed_src_pc_gt) ** 2, axis=1))

        # Target point loss
        transformed_tgt_pc_pred = tgt_rot_gt @ tgt_pc + tgt_trans_pred  # batch x 3 x 1024
        with torch.no_grad():
            transformed_tgt_pc_gt = tgt_rot_pred @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        # transformed_tgt_pc_pred = self.batch_quat2mat(tgt_quat_pred) @ tgt_pc + tgt_trans_pred # batch x 3 x 1024
        # transformed_tgt_pc_gt   = self.batch_quat2mat(tgt_quat_gt) @ tgt_pc + tgt_trans_gt     # batch x 3 x 1024
        tgt_point_loss = torch.mean(torch.sum((transformed_tgt_pc_pred - transformed_tgt_pc_gt) ** 2, axis=1))

        # Point loss
        point_loss = (src_point_loss + tgt_point_loss) / 2.0
        return point_loss

    def compute_trans_loss(self, batch_data, pred_data):
        # Ground truths
        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1

        # Model predictions
        src_trans_pred = pred_data['src_trans']  # batch x 3 x 1
        tgt_trans_pred = pred_data['tgt_trans']  # batch x 3 x 1

        src_trans_pred = src_trans_pred.unsqueeze(dim=2)
        tgt_trans_pred = tgt_trans_pred.unsqueeze(dim=2)

        # Compute loss
        src_trans_loss = F.l1_loss(src_trans_pred, src_trans_gt)
        tgt_trans_loss = F.l1_loss(tgt_trans_pred, tgt_trans_gt)
        trans_loss = (src_trans_loss + tgt_trans_loss) / 2.0
        return trans_loss

    def compute_rot_loss(self, batch_data, pred_data):
        # Ground truths
        src_R_6d = batch_data['src_rot']
        tgt_R_6d = batch_data['tgt_rot']

        src_R_6d_pred = pred_data['src_rot']
        tgt_R_6d_pred = pred_data['tgt_rot']

        src_rot_loss = torch.mean(utils.get_6d_rot_loss(src_R_6d, src_R_6d_pred))
        tgt_rot_loss = torch.mean(utils.get_6d_rot_loss(tgt_R_6d, tgt_R_6d_pred))
        rot_loss = (src_rot_loss + tgt_rot_loss) / 2.0
        # print(rot_loss)
        return rot_loss

    def compute_recon_loss(self, batch_data, pred_data):

        recon_pts = pred_data['recon_pts'] # batch x 1024 x 3

        src_pc = batch_data['src_pc'].float()  # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024
        src_quat_gt = batch_data['src_rot'].float()
        tgt_quat_gt = batch_data['tgt_rot'].float()

        src_Rs = utils.bgs(src_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3
        tgt_Rs = utils.bgs(tgt_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3

        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        with torch.no_grad():
            transformed_src_pc_gt = src_Rs @ src_pc + src_trans_gt  # batch x 3 x 1024
            transformed_tgt_pc_gt = tgt_Rs @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        gt_pts = torch.cat([transformed_src_pc_gt, transformed_tgt_pc_gt], dim=2).permute(0, 2, 1) # batch x 2048 x 3
        # set_trace()
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = self.chamLoss(gt_pts, recon_pts)
        recon_loss = torch.mean(dist1) + torch.mean(dist2)
        # torch.save(gt_pts[0],"./gt_pts.pt")
        # torch.save(recon_pts[0],"./recon_pts.pt")
        # torch.save(src_pc[0].permute(1,0),"./src_pc.pt")
        # torch.save(tgt_pc[0].permute(1,0),"./tgt_pc.pt")
        # set_trace()
        return recon_loss

    def recover_R_from_6d(self, R_6d):
        # R_6d is batch * 6
        R = utils.bgs(R_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        # R is batch * 3 * 3
        return R

    def quat_to_eular(self, quat):
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        r = R.from_quat(quat)
        euler0 = r.as_euler('xyz', degrees=True)

        return euler0

    def training_step(self, batch_data, batch_idx):
        self.iter_counts += 1
        total_loss, point_loss, rot_loss, trans_loss, recon_loss = self.forward_pass(batch_data, mode='train')
        return {"total_loss": total_loss,
                "point_loss": point_loss,
                "rot_loss": rot_loss,
                "trans_loss": trans_loss,
                "recon_loss": recon_loss
                }

    def validation_step(self, batch_data, batch_idx, vis_idx=-1):
        total_loss, point_loss, rot_loss, trans_loss, recon_loss = self.forward_pass(batch_data, mode='val', vis_idx=vis_idx)
        return {"total_loss": total_loss,
                "point_loss": point_loss,
                "rot_loss": rot_loss,
                "trans_loss": trans_loss,
                "recon_loss": recon_loss
                }

    def validation_metrics(self, batch_data, batch_idx, vis_idx=-1):
        GD, RMSE_R, RMSE_T, PA  = self.forward_pass(batch_data, mode='val', vis_idx=vis_idx)
        return GD, RMSE_R, RMSE_T, PA

    def test_step(self, batch_data, batch_idx):
        total_loss, point_loss, rot_loss, trans_loss = self.forward_pass(batch_data, mode='train')
        return total_loss
        # return {'loss': total_loss, 'log': tensorboard_logs}

    def calculate_metrics(self, batch_data, pred_data, mode):
        GD = self.compute_rot_loss(batch_data, pred_data)

        src_pc = batch_data['src_pc'].float()  # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024
        src_quat_gt = batch_data['src_rot'].float()
        tgt_quat_gt = batch_data['tgt_rot'].float()

        src_Rs = utils.bgs(src_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3
        tgt_Rs = utils.bgs(tgt_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3

        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        with torch.no_grad():
            transformed_src_pc_gt = src_Rs @ src_pc + src_trans_gt  # batch x 3 x 1024
            transformed_tgt_pc_gt = tgt_Rs @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        gt_pts = torch.cat([transformed_src_pc_gt, transformed_tgt_pc_gt], dim=2).permute(0, 2, 1) # batch x 2048 x 3

        pred_R_src = self.recover_R_from_6d(pred_data['src_rot'])
        pred_R_tgt = self.recover_R_from_6d(pred_data['tgt_rot'])
        pred_t_src = pred_data['src_trans'].view(-1, 3, 1)
        pred_t_tgt = pred_data['tgt_trans'].view(-1, 3, 1)


        gt_euler_src = pytorch3d.transforms.matrix_to_euler_angles(src_Rs, convention="XYZ")
        gt_euler_tgt = pytorch3d.transforms.matrix_to_euler_angles(tgt_Rs, convention="XYZ")

        pred_euler_src = pytorch3d.transforms.matrix_to_euler_angles(pred_R_src, convention="XYZ")
        pred_euler_tgt = pytorch3d.transforms.matrix_to_euler_angles(pred_R_tgt, convention="XYZ")

        diff_src_1 = (gt_euler_src - pred_euler_src).abs() * 180 / np.pi
        diff_src_2 = 360. - (gt_euler_src - pred_euler_src).abs() * 180 / np.pi
        diff_src = torch.minimum(diff_src_1, diff_src_2)
        RMSE_src = diff_src.pow(2).mean(dim=-1)**0.5
        diff_tgt_1 = (gt_euler_tgt - pred_euler_tgt).abs() * 180 / np.pi
        diff_tgt_2 = 360. - (gt_euler_tgt - pred_euler_tgt).abs() * 180 / np.pi
        diff_tgt = torch.minimum(diff_tgt_1, diff_tgt_2)
        RMSE_tgt = diff_tgt.pow(2).mean(dim=-1)**0.5
        RMSE_R = (RMSE_src + RMSE_tgt) / 2

        with torch.no_grad():
            transformed_src_pc_pred = pred_R_src @ src_pc + pred_t_src  # batch x 3 x 1024
            transformed_tgt_pc_pred = pred_R_tgt @ tgt_pc + pred_t_tgt  # batch x 3 x 1024
        recon_pts = torch.cat([transformed_src_pc_pred, transformed_tgt_pc_pred], dim=2).permute(0, 2, 1) # batch x 2048 x 3

        dist1, dist2, idx1, idx2 = self.chamLoss(gt_pts, recon_pts)
        PA = torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)

        thre = 0.02
        acc = (PA < thre)
        # the official code is doing avg per-shape acc (not per-part)
        PA = acc.sum(-1) / acc.shape[0]

        RMSE_T_1 = (pred_t_src - src_trans_gt).pow(2).mean(dim=-1) ** 0.5
        RMSE_T_2 = (pred_t_tgt - tgt_trans_gt).pow(2).mean(dim=-1) ** 0.5
        RMSE_T = (RMSE_T_1 + RMSE_T_2) / 2

        return GD, RMSE_R, RMSE_T, PA

    def forward_pass(self, batch_data, mode, vis_idx=-1):

        pred_data = self.forward(batch_data['src_pc'].float(), batch_data['tgt_pc'].float())
        self.check_network_property(batch_data, pred_data)
        if self.cfg.model.point_loss:
            point_loss = self.compute_point_loss(batch_data, pred_data)
        else:
            point_loss = 0.0

        # Pose loss
        rot_loss = self.compute_rot_loss(batch_data, pred_data)
        trans_loss = self.compute_trans_loss(batch_data, pred_data)
        if self.cfg.model.recon_loss:
            recon_loss = self.compute_recon_loss(batch_data, pred_data)
        else:
            recon_loss = 0.0

        if vis_idx > 0:
            debug_vis(batch_data, self.cfg, pred_data, vis_idx)
            debug_vis_gt(batch_data, self.cfg, pred_data, vis_idx)

        if mode == 'val':
            return self.calculate_metrics(batch_data, pred_data, mode)
        # Total loss
        total_loss = point_loss + rot_loss + trans_loss + recon_loss

        return total_loss, point_loss, rot_loss, trans_loss, recon_loss

