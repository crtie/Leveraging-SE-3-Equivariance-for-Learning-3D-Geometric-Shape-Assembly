import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn.functional as F

def render_pts_label_png(fn, pc, color):
    # pc: (num_points, 3), color: (num_points,)
    new_color = []
    for i in range(len(color)):
        if color[i] == 1:
            new_color.append('#ab4700')
        else:
            new_color.append('#00479e')
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    # 标题
    plt.title('point cloud')
    # 利用xyz的值，生成每个点的相应坐标（x,y,z）
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=new_color, marker='.', s=5, linewidth=0, alpha=1)
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 显示
    plt.savefig(fn+'.png')

def compute_distance_between_rotations(P,Q):
    #! input two rotation matrices, output the distance between them (unit is rad)
    #! P,Q are 3x3 numpy arrays
    P = np.asarray(P)
    Q = np.asarray(Q)
    R = np.matmul(P,Q.swapaxes(1,2))
    theta = np.arccos(np.clip((np.trace(R,axis1 = 1,axis2 = 2) - 1)/2,-1,1))
    return np.mean(theta)

def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)

def get_6d_rot_loss(pred_6d, gt_6d):
    #input pred_6d , gt_6d : batch * 2 * 3
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    return theta


def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')
