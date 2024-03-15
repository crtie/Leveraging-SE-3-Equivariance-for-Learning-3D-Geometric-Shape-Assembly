import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import pytorch_lightning as pl

from pytorch3d.transforms import quaternion_to_matrix
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
from pdb import set_trace
from multi_part_assembly.utils import (
    _get_clones,
    chamfer_distance,
    _valid_mean,
    rot_pc,
    transform_pc,
    Rotation3D,
    shape_cd_loss,
)
from multi_part_assembly.models import BaseModel
from multi_part_assembly.models import build_encoder, StocasticPoseRegressor
from pointnet2_ops import pointnet2_utils
from .modules import *
from .dgcnn import DGCNN_New
from .utils import *
from pdb import set_trace
from scipy.spatial.transform import Rotation as R


class VNNModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.encoder = self.init_encoder()
        self.pose_predictor = self.init_pose_predictor()

        self.R = torch.tensor(
            [
                [0.26726124, -0.57735027, 0.77151675],
                [0.53452248, -0.57735027, -0.6172134],
                [0.80178373, 0.57735027, 0.15430335],
            ],
            dtype=torch.float64,
        ).unsqueeze(0)
        self.close_eps = self.cfg.model.close_eps
        self.iters = 0
        self.flag = True

    def init_encoder(self):
        if self.cfg.model.encoder == "dgcnn":
            encoder = DGCNN_New(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoder == "vn_dgcnn":
            encoder = VN_DGCNN(feat_dim=self.cfg.model.pc_feat_dim)
        return encoder

    def init_pose_predictor(self):
        if self.cfg.model.regressor == "original":
            pose_predictor = Ori_Regressor(pc_feat_dim=self.cfg.model.pc_feat_dim)
        if self.cfg.model.regressor == "vnn":
            pose_predictor = VN_Regressor(pc_feat_dim=self.cfg.model.pc_feat_dim)
        return pose_predictor

    def init_discriminator(self):
        return Discriminator(num_points=self.cfg.data.num_pc_points)
        # return Discriminator()

    def check_equiv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(torch.matmul(x, R) - xR))
        if mean_diff > self.close_eps:
            print(f"---[Equiv check]--- {name}: {mean_diff}")
        return

    def check_inv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(x - xR))
        if mean_diff > self.close_eps:
            print(f"---[Equiv check]--- {name}: {mean_diff}")
        return

    def check_network_property(self, gt_data, pred_data):
        with torch.no_grad():
            B, P, N, _ = gt_data["part_pcs"].shape
            pcs = gt_data["part_pcs"].reshape(B * P, N, 3)
            R = self.R.float().repeat(B * P, 1, 1).to(pcs.device)
            pcs_R = torch.matmul(pcs, R).reshape(B, P, N, 3)
            data_R = copy.deepcopy(gt_data)
            data_R["part_pcs"] = pcs_R
            pred_data_R = self.forward(data_R)
            #! check pointcloud equivariance (should be naturally equivariant)
            self.check_equiv(pcs, R, pcs_R.reshape(B * P, N, 3), "pcs")

            #! check equivariance of equivariant features
            equiv_feats = pred_data["equiv_feats"].reshape(B * P, -1, 3)
            equiv_feats_R = pred_data_R["equiv_feats"].reshape(B * P, -1, 3)
            self.check_equiv(equiv_feats, R, equiv_feats_R, "equiv_feats")

            equiv_R = pred_data["rot"].reshape(B * P, 3, 3)
            equiv_R_R = pred_data_R["rot"].reshape(B * P, 3, 3)
            # self.check_equiv(equiv_R, R, equiv_R_R, 'rotation')

            #! check invariance of invariant features
            inv_feats = pred_data["inv_feats"]
            c = inv_feats.shape[2]
            inv_feats = inv_feats.reshape(B * P, c, c)
            inv_feats_R = pred_data_R["inv_feats"].reshape(B * P, c, c)

            trans = pred_data["trans"].reshape(B * P, 3)
            trans_R = pred_data_R["trans"].reshape(B * P, 3)
            # self.check_inv(trans, R, trans_R, 'translation')
            self.check_inv(inv_feats, R, inv_feats_R, "inv_feats")
        return

    def _extract_part_feats(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        # part valid (B, P)
        B, P, _, N = part_pcs.shape  # [B, P, 3, N]
        valid_mask = part_valids == 1
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, 3, N]
        valid_feats_equiv, valid_feats_inv = self.encoder(valid_pcs)

        equiv_pc_feats = torch.zeros(B, P, self.cfg.model.pc_feat_dim * 2, 3).type_as(
            valid_feats_equiv
        )
        equiv_pc_feats[valid_mask] = valid_feats_equiv

        inv_pc_feats = torch.zeros(
            B, P, self.cfg.model.pc_feat_dim * 2, self.cfg.model.pc_feat_dim * 2
        ).type_as(valid_feats_inv)
        inv_pc_feats[valid_mask] = valid_feats_inv
        return equiv_pc_feats, inv_pc_feats

    def _extract_total_feats(self, batch_data):
        part_pcs = batch_data["part_pcs"]
        part_valids = batch_data["part_valids"]
        B, P, _, N = part_pcs.shape  # [B, P, 3, N]

        # Ground truths
        rot_gt = batch_data["part_rot"].to_rmat()
        trans_gt = batch_data["part_trans"].float()

        total_pt = transform_pc(trans_gt, rot_gt, part_pcs, rot_type="rmat").reshape(
            B, -1, 3
        )
        idx = pointnet2_utils.furthest_point_sample(
            total_pt[:, :, :3].contiguous(), self.cfg.data.num_pc_points
        ).long()
        idx = idx.view(*idx.shape, 1).repeat_interleave(total_pt.shape[-1], dim=2)
        sampled_points = torch.gather(total_pt, dim=1, index=idx)
        total_equiv_feats, total_inv_feats = self.encoder(
            sampled_points.permute(0, 2, 1)
        )

        return torch.bmm(
            total_inv_feats, total_equiv_feats
        )  # (B, self.cfg.model.num_feats * 2, 3)

    def _predict_pose(self, equiv_feats, part_valids):
        """Predict per-part poses."""
        # part valid (B, P)
        # equiv_feats (B, P, 2*feat_dim, 3)
        B, P, C, _ = equiv_feats.shape
        valid_mask = part_valids == 1
        valid_equiv_feats = equiv_feats[valid_mask]  # [n, 2*feat_dim, 3]
        valid_R_6d, valid_trans = self.pose_predictor(
            valid_equiv_feats
        )  # [n, 2, 3], [n, 3]
        R_6d = torch.zeros(B, P, 2, 3).type_as(valid_R_6d)
        R_6d[valid_mask] = valid_R_6d
        trans = torch.zeros(B, P, 3).type_as(valid_trans)
        trans[valid_mask] = valid_trans

        return R_6d, trans

    def _recon_pts(self, inv_feats, part_valids):
        # inv_feats (B, P, 2*feat_dim, 2*feat_dim)
        # part_valids (B, P)
        B, P, C, _ = inv_feats.shape
        valid_mask = part_valids == 1
        valid_inv_feats = inv_feats[valid_mask]  # [n, 2*feat_dim, 2*feat_dim]
        global_inv_feats = torch.sum(
            inv_feats, dim=1, keepdims=False
        )  # [B, 2*feat_dim, 2*feat_dim]
        return global_inv_feats  # (batch_size, num_parts, N, 3), (batch_size, N, 3), (batch_size, 2*feat_dim, 2*feat_dim)

    def forward(self, batch_data):
        part_pcs = batch_data["part_pcs"]
        B, P, _, _ = part_pcs.shape  # [B, P, N, 3]
        part_valids = batch_data["part_valids"]
        part_pcs = part_pcs.permute(0, 1, 3, 2)
        #! calc equiv and inv feats for each part
        equiv_feats, inv_feats = self._extract_part_feats(
            part_pcs, part_valids
        )  # (batch_size, num_parts, 2*feat_dim, 3), (batch_size, num_parts, 2*feat_dim, 2*feat_dim)
        #! calc global inv feats
        global_inv_feats = self._recon_pts(inv_feats, part_valids)
        global_inv_feats = global_inv_feats.unsqueeze(1).repeat(1, P, 1, 1)

        # (batch_size,2*feat_dim, 2*feat_dim)
        #! calc global equiv feats arter correlation
        if self.cfg.model.with_corr:
            GF = torch.bmm(
                global_inv_feats.reshape(
                    -1, self.cfg.model.pc_feat_dim * 2, self.cfg.model.pc_feat_dim * 2
                ),
                equiv_feats.reshape(-1, self.cfg.model.pc_feat_dim * 2, 3),
            ).reshape(
                B, P, -1, 3
            )  # (batch_size, num_parts, 2*feat_dim, 3)
            # print(global_inv_feats.shape)
        else:
            GF = equiv_feats

        R_6d, trans = self._predict_pose(
            equiv_feats, part_valids
        )  # (batch_size, num_parts, 2, 3), (batch_size, num_parts, 3)
        R_9d = self.recover_R_from_6d(R_6d)
        pred_dict = {
            "equiv_feats": GF,
            "part_F": equiv_feats,
            "inv_feats": inv_feats,
            "rot": R_9d,
            "trans": trans,
            # 'recon_pts': recon_pts,
            # 'whole_recon_pcs': whole_recon_pcs,
        }

        if self.cfg.model.translation_embedding == True:
            total_equiv_feats = self._extract_total_feats(batch_data)
            pred_dict["total_equiv_feats"] = total_equiv_feats
        return pred_dict

    def compute_point_loss(self, batch_data, pred_data, mode):
        part_pcs = batch_data["part_pcs"]
        valids = batch_data["part_valids"]
        # Ground truths
        rot_gt = batch_data["part_rot"]
        trans_gt = batch_data["part_trans"].float()

        # Model predictions
        rot_pred = Rotation3D(pred_data["rot"].float(), rot_type="rmat")
        trans_pred = pred_data["trans"].float()
        if self.cfg.model.pointloss == "L2":
            pc_gt = transform_pc(trans_gt, rot_gt, part_pcs)
            pc_pred = transform_pc(trans_pred, rot_pred, part_pcs)
            loss_per_data = (pc_gt - pc_pred).pow(2).sum(-1).mean(-1)  # [B, P]
            loss_per_data = _valid_mean(loss_per_data, part_valids)
            return loss_per_data
        elif self.cfg.model.pointloss == "cham":
            transform_pt_cd_loss, pred_pts, gt_pts = shape_cd_loss(
                part_pcs, trans_pred, trans_gt, rot_pred, rot_gt, valids, ret_pts=True
            )
            return transform_pt_cd_loss, pred_pts, gt_pts

    def compute_trans_loss(self, batch_data, pred_data):
        # Ground truths
        trans_gt = batch_data["part_trans"].float()  # batch x num_parts x 3
        # Model predictions
        trans_pred = pred_data["trans"]  # batch x num_parts x 3
        # Compute loss
        part_valids = batch_data["part_valids"]
        trans_loss = (trans_gt - trans_pred).pow(2).sum(-1)
        trans_loss = _valid_mean(trans_loss, part_valids)
        return trans_loss

    def compute_rot_loss(self, batch_data, pred_data):
        R_9d = batch_data["part_rot"].to_rmat()
        part_valids = batch_data["part_valids"]
        B, P = part_valids.shape[:2]
        R_9d_pred = pred_data["rot"]
        rot_loss = new_get_6d_rot_loss(R_9d, R_9d_pred, self.cfg.model.rot_loss)
        rot_loss = _valid_mean(rot_loss, part_valids)
        return rot_loss

    def compute_recons_loss(self, batch_data, pred_data):
        recons_pts = pred_data["recon_pts"]  # (batch_size, num_parts, N, 3)
        whole_recon_pcs = pred_data["whole_recon_pcs"]  # (batch_size, N, 3)
        part_pcs = batch_data["part_pcs"]
        B, P, N, _ = part_pcs.shape  # [B, P, N, 3]
        recon_pcs = recons_pts.reshape(B * P, N, 3)

        part_valids = batch_data["part_valids"]

        # Ground truths
        rot_gt = batch_data["part_rot"].to_rmat()
        trans_gt = batch_data["part_trans"].float()

        transformed_pc_gt = transform_pc(
            trans_gt, rot_gt, part_pcs, rot_type="rmat"
        ).reshape(B * P, N, 3)
        dist1, dist2 = chamfer_distance(transformed_pc_gt, recon_pcs)

        whole_pc_gt = transformed_pc_gt.reshape(B, P, N, 3).reshape(B, -1, 3)
        dist3, dist4 = chamfer_distance(whole_pc_gt, whole_recon_pcs)
        part_cham_loss = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        part_cham_loss = part_cham_loss.view(B, -1).type_as(recon_pcs)  # [B, P]
        part_cham_loss = _valid_mean(part_cham_loss, part_valids)
        whole_cham_loss = torch.mean(dist3, dim=1) + torch.mean(dist4, dim=1)  # [B]
        cham_loss = part_cham_loss + whole_cham_loss
        return cham_loss

    def compute_embedding_loss(self, batch_data, pred_data):
        total_equiv_feats = pred_data[
            "total_equiv_feats"
        ]  # (batch_size, 2*self.cfg.data.feat_dim, 3)
        part_equiv_feats = pred_data[
            "part_F"
        ]  # (batch_size, num_parts, 2*self.cfg.data.feat_dim, 3)
        B, P = part_equiv_feats.shape[:2]
        pred_total_equiv_feats = torch.sum(
            part_equiv_feats, dim=1
        )  # (batch_size, 2*self.cfg.data.feat_dim, 3)
        embedding_loss = torch.mean(
            (abs(total_equiv_feats - pred_total_equiv_feats)), dim=(1, 2)
        )
        return embedding_loss

    def compute_adv_loss(self, batch_data, pred_data, pred_pts, gt_pts):
        #! this function is used only when training with adversarial loss
        # pred_pts, gt_pts: (B, P, N, 3)
        B, P, N, _ = pred_pts.shape
        total_gt = gt_pts.reshape(B, -1, 3)
        idx1 = pointnet2_utils.furthest_point_sample(
            total_gt[:, :, :3].contiguous(), self.cfg.data.num_pc_points
        ).long()
        idx1 = idx1.view(*idx1.shape, 1).repeat_interleave(total_gt.shape[-1], dim=2)
        sampled_points_gt = torch.gather(total_gt, dim=1, index=idx1)

        total_pred = gt_pts.reshape(B, -1, 3)
        idx2 = pointnet2_utils.furthest_point_sample(
            total_pred[:, :, :3].contiguous(), self.cfg.data.num_pc_points
        ).long()
        idx2 = idx2.view(*idx2.shape, 1).repeat_interleave(total_pred.shape[-1], dim=2)
        sampled_points_pred = torch.gather(total_pred, dim=1, index=idx2)
        pg_label = self.discriminator(sampled_points_gt.permute(0, 2, 1)).squeeze(
            1
        )  # (bs, 1)
        pp_label = self.discriminator(sampled_points_pred.permute(0, 2, 1)).squeeze(
            1
        )  # (bs, 1)

        gp_label = torch.zeros(B).to(pp_label.device)
        gg_label = torch.ones(B).to(pp_label.device)
        #! label is 0 for fake and 1 for real

        adv_loss_G = self.advLoss(pp_label, gg_label)
        adv_loss_D = self.advLoss(pg_label, gg_label) + self.advLoss(pp_label, gp_label)
        acc_p = (pp_label < 0.5).float()
        acc_g = (pg_label > 0.5).float()
        return adv_loss_G, adv_loss_D, acc_p, acc_g

    def check_recover(self):
        bs = 32
        R = self.R.cpu().float()
        R_6d = F.normalize(torch.rand(bs, 2, 3), p=2, dim=2)
        R_6d_R = R_6d.bmm(R)

        # self.check_equiv(R_6d.to(self.R.device), R_6d_R.to(self.R.device), 'Check Recover 6d')

        R_9d = self.recover_R_from_6d(R_6d)
        R_9d_R = self.recover_R_from_6d(R_6d_R)

        self.check_equiv(
            R_9d.to(self.R.device), R_9d_R.to(self.R.device), "Check Recover 9d"
        )

    def recover_R_from_6d(self, R_6d):
        # R_6d: (bs, P, 2, 3) or (bs, P, 6)
        B = R_6d.shape[0]
        P = R_6d.shape[1]
        R_6d = R_6d.reshape(-1, 2, 3)
        R = bgs(R_6d.reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
        R = R.reshape(B, P, 3, 3)
        return R

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1, mode="train"):
        pred_data = self.forward(data_dict)
        self.iters += 1
        if self.cfg.model.check_equiv == True:
            self.check_network_property(data_dict, pred_data)

        # point_loss, pred_pts, gt_pts = self.compute_point_loss(data_dict, pred_data, self.cfg.model.point_loss)
        rot_loss = self.compute_rot_loss(data_dict, pred_data)
        trans_loss = self.compute_trans_loss(data_dict, pred_data)
        # recons_loss = self.compute_recons_loss(data_dict, pred_data)
        loss_dict = {}
        # loss_dict['point_loss'] = point_loss
        loss_dict["rot_loss"] = rot_loss
        loss_dict["trans_loss"] = trans_loss
        # loss_dict['recons_loss'] = recons_loss
        if self.cfg.model.with_adv:
            adv_loss_G, adv_loss_D, acc_p, acc_g = self.compute_adv_loss(
                data_dict, pred_data, pred_pts, gt_pts
            )
            acc = (torch.mean(acc_g) + torch.mean(acc_p)) / 2.0
            if mode == "val":
                loss_dict["adv_G_loss"] = adv_loss_G
                loss_dict["adv_D_loss"] = adv_loss_D
            else:
                # print(optimizer_idx)
                # print(f'[train]accuracy of prediction ',torch.mean(acc_p))
                # print(f'[train]accuracy of gt ',torch.mean(acc_g))
                # if self.iters > 1:
                #     for key in self.encoder.state_dict().keys():
                #         if self.encoder.state_dict()[key].equal(self.unchange_param[key]) == False:
                #             print(f'encoder {key} changed')
                #     for key in self.discriminator.state_dict().keys():
                #         if self.discriminator.state_dict()[key].equal(self.change_param[key]) == False:
                #             print(f'discriminator {key} changed')
                # self.change_param = self.discriminator.state_dict()
                # self.unchange_param = self.encoder.state_dict()
                if optimizer_idx == 0:  #! idx 0 is for encoder
                    loss_dict["adv_G_loss"] = adv_loss_G
                elif optimizer_idx == 1:  #! idx 1 is for discriminator
                    loss_dict = {}
                    loss_dict["adv_D_loss"] = adv_loss_D
            loss_dict["adv_accuracy"] = acc.unsqueeze(0).repeat(adv_loss_G.shape[0])

        #! all terms in loss_dict should be (B)

        if self.cfg.model.translation_embedding:
            embedding_loss = self.compute_embedding_loss(data_dict, pred_data)
            loss_dict["embedding_loss"] = embedding_loss
        if not self.training:
            pred_data["rot"] = Rotation3D(pred_data["rot"], rot_type="rmat")
            eval_dict = self._calc_metrics(
                data_dict, pred_data, data_dict["part_trans"], data_dict["part_rot"]
            )
            loss_dict.update(eval_dict)
        return loss_dict, pred_data
