import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

import pytorch_lightning as pl

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (idx + idx_base).view(-1)

    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  
                                        # -> (batch_size*num_points, num_dims) 
                                        #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(pl.LightningModule):

    def __init__(self, feat_dim):

        super().__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(feat_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()                 # x:      batch x   3 x num of points
        x      = get_graph_feature(x)                               # x:      batch x   6 x num of points x 20

        x1     = self.conv1(x)                                      # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]                    # x1_max: batch x  64 x num of points x 1

        x2     = self.conv2(x1)                                     # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]                    # x2_max: batch x  64 x num of points x 1

        x3     = self.conv3(x2)                                     # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]                    # x3_max: batch x 128 x num of points x 1

        x4     = self.conv4(x3)                                     # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]                    # x4_max: batch x 256 x num of points x 1
 
        x_max  = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1) # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)        # point feat:  batch x 512 x num of points

        return point_feat


class DGCNN_New(pl.LightningModule):

    def __init__(self, feat_dim):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(feat_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear0 = nn.Linear(256, 3)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()  # x:      batch x   3 x num of points
        x = get_graph_feature(x)  # x:      batch x   6 x num of points x 20

        x1 = self.conv1(x)  # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]  # x1_max: batch x  64 x num of points x 1

        x2 = self.conv2(x1)  # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]  # x2_max: batch x  64 x num of points x 1

        x3 = self.conv3(x2)  # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]  # x3_max: batch x 128 x num of points x 1

        x4 = self.conv4(x3)  # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]  # x4_max: batch x 256 x num of points x 1

        x_max = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1)  # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)  # point feat:  batch x 512 x num of points
        final_point_feat = torch.cat([point_feat, point_feat], dim=1)  # final_point_feat:  bs, 1024, np
        final_point_feat = final_point_feat.permute(0, 2, 1)  # bs, np, 1024

        point_feat = point_feat.permute(0, 2, 1)  # bs, np, 512
        point_feat = point_feat.reshape(batch_size*num_points, 256)  # bs*np, 512
        f_feat = self.linear0(point_feat)  # f_feat: bs*np, 3
        f_feat = f_feat.reshape(batch_size, num_points, 3)
        return f_feat, final_point_feat

class DGCNN_cls(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(128 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 32)
        self.linear5 = nn.Linear(32, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, 128)
        x = self.linear4(x)  # (batch_size, 128) -> (batch_size, 32)
        x = self.linear5(x)  # (batch_size, 32) -> (batch_size, 1)

        return x