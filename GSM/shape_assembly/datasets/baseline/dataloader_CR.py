import os
import sys
import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import trimesh
from PIL import Image
import json
from progressbar import ProgressBar
# from pyquaternion import Quaternion
import random
import copy
import time
import ipdb
from scipy.spatial.transform import Rotation as R
from pdb import set_trace

os.environ['PYOPENGL_PLATFORM'] = 'egl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))

from pytorch3d.transforms import quaternion_to_matrix
from mesh_to_sdf import sample_sdf_near_surface

## pc, quat, trans of (A, B)

def load_data(file_dir, cat_shape_dict):
    with open(file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            cat_shape_dict[cat].append(shape_id)
    return cat_shape_dict


class ShapeAssemblyDataset(data.Dataset):

    def __init__(self, data_root_dir, data_csv_file, data_features=[], num_points = 1024 ,num_query_points = 1024 ,data_per_seg = 1):
        self.data_root_dir = data_root_dir
        self.data_csv_file = data_csv_file
        self.num_points = num_points
        self.num_query_points = num_query_points
        self.data_features = data_features
        self.data_per_seg = data_per_seg
        self.dataset = []

        # currently, only consider the ShapeNet dataset
        self.category_list = ['02773838', '02880940', 'Box', 'Hat', '03593526', '03797390',
                              'Plate', 'Shoe', '04256520', '04379243', 'Toy']
        self.category_dict = {'02773838': 'Bag', '02880940': 'Bowl', 'Box': 'Box', 'Hat': 'Hat', '03593526': 'Jar', '03797390': 'Mug',
                              'Plate': 'Plate', 'Shoe': 'Shoe', '04256520': 'Sofa', '04379243': 'Table', 'Toy': 'Toy'}

        self.cut_list = ["planar", "parabolic", "sine", "square", "pulse"]


        self.category_cnt_dict = {}
        for category_id in self.category_list:
            self.category_cnt_dict[category_id] = 0

        self.cut_cnt_dict = {}
        for cut_name in self.cut_list:
            self.cut_cnt_dict[cut_name] = 0


    def transform_pc_to_rot(self, pcs):
        # zero-centered
        pc_center = (pcs.max(axis=0, keepdims=True) + pcs.min(axis=0, keepdims=True)) / 2
        pc_center = pc_center[0]
        new_pcs = pcs - pc_center

        def bgs(d6s):
            bsz = d6s.shape[0]
            b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
            a2 = d6s[:, :, 1]
            b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
            b3 = torch.cross(b1, b2, dim=1)
            return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

        # randomly sample two rotation matrices
        rotmat = bgs(torch.rand(1, 6).reshape(-1, 2, 3).permute(0, 2, 1))
        new_pcs = (rotmat.reshape(3, 3) @ new_pcs.T).T

        gt_rot = rotmat[:, :, :2].permute(0, 2, 1).reshape(6).numpy()

        # quat = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])
        # quat = np.array([ 0.13961374, -0.2859178, 0.37262626, 0.87172741])
        # quat = quat / np.linalg.norm(quat)
        # r = R.from_quat(quat)
        # rotation_matrix = r.as_matrix()
        # print('rotation_matric: ', rotation_matrix)
        # new_pcs = (rotation_matrix @ new_pcs.T).T

        # quat_inv = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        # pc_center = np.array([0.03022753, 0.03218459, -0.2821579])
        return new_pcs, pc_center, gt_rot


    def load_data(self):
        # currently, only consider the ShapeNet dataset
        # print('load once')
        cat_shape_dict = {}
        for category_id in self.category_list:
            cat_shape_dict[self.category_dict[category_id]] = []
        cat_shape_dict = load_data(self.data_csv_file, cat_shape_dict)

        bar = ProgressBar()

        for category_i in bar(range(len(self.category_list))):
            category_id = self.category_list[category_i]
            for shape_id in cat_shape_dict[self.category_dict[category_id]]:
                cur_dir = os.path.join(self.data_root_dir, category_id, shape_id)
                total_mesh_file = os.path.join(cur_dir, 'solid/model_normalized_watertight_fix.obj')
                # query_points_file_total = os.path.join(cur_dir, 'solid/query_points_total.npy')
                # sdf_file_total = os.path.join(cur_dir, 'solid/sdf_total.npy')
                for root, dirs, files in os.walk(cur_dir):
                    for dir in dirs:
                        solid = os.path.join(root, dir)
                        file_list = os.listdir(solid)
                        for cut_type in file_list:
                            cut = os.path.join(solid, cut_type)
                            if os.path.isdir(cut) == False:
                                continue
                            cut_list = os.listdir(cut)
                            for instance_id in cut_list:
                                instance_dir = os.path.join(cut, instance_id)
                                fileA = os.path.join(instance_dir, 'partA.npy')
                                fileB = os.path.join(instance_dir, 'partB.npy')
                                mesh_file_A = os.path.join(instance_dir, 'partA.obj')
                                mesh_file_B = os.path.join(instance_dir, 'partB.obj')
                                # query_points_file_A = os.path.join(root, dir, 'query_points_A.npy')
                                # query_points_file_B = os.path.join(root, dir, 'query_points_B.npy')
                                # sdf_file_A = os.path.join(root, dir, 'sdf_A.npy')
                                # sdf_file_B = os.path.join(root, dir, 'sdf_B.npy')
                                if not os.path.exists(fileA) or not os.path.exists(fileB):
                                    continue

                                category_name = self.category_dict[category_id]
                                self.category_cnt_dict[category_id] += 1
                                cut_name = fileA.split('/')[-3]
                                self.cut_cnt_dict[cut_name] += 1

                                result_id = fileA.split('/')[-2]

                                # print('okay_fileA: ', fileA)
                                gt_pcs_A = np.load(fileA)
                                gt_pcs_B = np.load(fileB)
                                for i in range(self.data_per_seg):
                                    gt_pcs_total = np.concatenate((gt_pcs_A, gt_pcs_B), axis=0)
                                    per = np.random.permutation(gt_pcs_total.shape[0])
                                    gt_pcs_total = gt_pcs_total[per]
                                    # quan @ (gt_pcs - trans) = new_pcs
                                    # gt_pcs = quan_inv @ new_pcs + trans

                                    # new_pcs_A, sample_points_A, trans_A, quat_A = self.transform_pc_and_query_points(gt_pcs_A, sample_points_A)
                                    # new_pcs_B, sample_points_B, trans_B, quat_B = self.transform_pc_and_query_points(gt_pcs_B, sample_points_B)
                                    # new_pcs_A, trans_A, quat_A = self.transform_pc(gt_pcs_A)
                                    # new_pcs_B, trans_B, quat_B = self.transform_pc(gt_pcs_B)

                                    self.dataset.append([None, None, None, mesh_file_A, fileA,
                                                        None, None, None, mesh_file_B, fileB,
                                                        gt_pcs_total, total_mesh_file,
                                                        category_name, shape_id, cut_name, result_id])


    def __str__(self):
        strout = 'category_cnts:\n'
        for category_id in self.category_list:
            strout += '%s: %d\n' % (self.category_dict[category_id], self.category_cnt_dict[category_id])
        strout += '\ncut_cnts:\n'
        for cut_name in self.cut_list:
            strout += '%s: %d\n' % (cut_name, self.cut_cnt_dict[cut_name])
        return strout

    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):
        # new_pcs_A, quat_A, trans_A, gt_mesh_A, sample_points_A, sdf_A, \
        # new_pcs_B, quat_B, trans_B, gt_mesh_A, sample_points_B, sdf_B, \
        # gt_pcs_total, gt_mesh_total, sample_points_total, sdf_total, \
        # category_name, shape_id, cut_name, result_id = self.dataset[index]

        flag = 0
        while flag == 0:
            _, _, _, mesh_file_A, point_fileA, \
            _, _, _, mesh_file_B, point_fileB ,\
            gt_pcs_total, total_mesh_file, \
            category_name, shape_id, cut_name, result_id = self.dataset[index]

            gt_pcs_A = np.load(point_fileA)
            gt_pcs_B = np.load(point_fileB)

            if gt_pcs_A[0][0] != 0 and gt_pcs_A[0][1] != 0:
                # 如果不是0，那么可行
                flag = 1
            else:
                # 如果是，就尝试下一个
                index += 1

        if gt_pcs_A[0][0] == 0 and gt_pcs_A[0][1] == 0:
            raise ValueError('getitem Zero encountered!')


        new_pcs_A, trans_A, rot_A = self.transform_pc_to_rot(gt_pcs_A)
        new_pcs_B, trans_B, rot_B = self.transform_pc_to_rot(gt_pcs_B)


        data_feats = dict()
        for feat in self.data_features:
            if feat == 'src_pc':
                data_feats['src_pc'] = new_pcs_A.T

            elif feat == 'tgt_pc':
                data_feats['tgt_pc'] = new_pcs_B.T

            elif feat == 'total_pc':
                data_feats['total_pc'] = gt_pcs_total.T

            elif feat == 'src_rot':
                data_feats['src_rot'] = rot_A

            elif feat == 'tgt_rot':
                data_feats['tgt_rot'] = rot_B

            elif feat == 'src_trans':
                data_feats['src_trans'] = trans_A.reshape(1, 3).T
                # print('src_trans: ',data_feats['src_trans'])

            elif feat == 'tgt_trans':
                data_feats['tgt_trans'] = trans_B.reshape(1, 3).T
                # print('tgt_trans: ',data_feats['tgt_trans'])

            # elif feat == 'src_sample_points':
            #     data_feats['src_sample_points'] = sample_points_A.T
            
            # elif feat == 'tgt_sample_points':
            #     data_feats['tgt_sample_points'] = sample_points_B.T

            # elif feat == 'total_sample_points':
            #     data_feats['total_sample_points'] = sample_points_total.T

            elif feat == 'src_mesh':
                data_feats['src_mesh'] = mesh_file_A
            
            elif feat == 'tgt_mesh':
                data_feats['tgt_mesh'] = mesh_file_B
            
            elif feat == 'total_mesh':
                data_feats['total_mesh'] = total_mesh_file

            # elif feat == 'src_sdf':
            #     data_feats['src_sdf'] = sdf_A.reshape(1, -1)
            
            # elif feat == 'tgt_sdf':
            #     data_feats['tgt_sdf'] = sdf_B.reshape(1, -1)

            # elif feat == 'total_sdf':
            #     data_feats['total_sdf'] = sdf_total.reshape(1, -1)

            elif feat == 'category_name':
                data_feats['category_name'] = category_name

            elif feat == 'shape_id':
                data_feats['shape_id'] = shape_id

            elif feat == 'cut_name':
                data_feats['cut_name'] = cut_name

            elif feat == 'result_id':
                data_feats['result_id'] = result_id
        # for key in data_feats.keys():
        #     print(key, data_feats[key].shape)
        # exit()
        return data_feats












