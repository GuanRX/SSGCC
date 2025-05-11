'''
# Contrastive Loss
'''
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .kmeans_gpu import kmeans
from utils.postprocess import kmeans_clustering_torch
from .attention import Attention
from sklearn.cluster import KMeans, SpectralClustering
from .ct_loss import InstanceLoss
from typing import  Literal, Tuple


@torch.no_grad()
def distributed_sinkhorn(out):
    epsilon = 0.05  # 可调
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * 1  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def comprehensive_similarity(Z1, Z2):
    Z = torch.cat([Z1, Z2], dim=0)
    # calculate the similarity matrix
    Z1_Z2 = torch.matmul(Z, Z.T)

    # Z1_Z2 = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
    #                    torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)

    return Z1_Z2

def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance

def high_confidence(Z, center, tau=0.9):
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tau)))
    index = torch.where(distance_norm <= value[-1],
                                torch.ones_like(distance_norm), torch.zeros_like(distance_norm))

    high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
    high_conf_index_v2 = high_conf_index_v1 + Z.shape[0]
    H = torch.cat([high_conf_index_v1, high_conf_index_v2], dim=0)
    H_mat = np.ix_(H.cpu(), H.cpu())
    return H, H_mat


def pseudo_matrix(P, S, node_num, beta):
    P = torch.cat([P, P], dim=0)
    Q = (P == P.unsqueeze(1)).float().to(P.device)
    S_norm = (S - S.min()) / (S.max() - S.min())
    M_mat = torch.abs(Q - S_norm) ** beta
    M = torch.cat([torch.diag(M_mat, node_num), torch.diag(M_mat, -node_num)], dim=0)
    return M, M_mat


def hard_sample_aware_infoNCE(S, M, pos_neg_weight, pos_weight, node_num):
    M = M.to('cuda')
    pos_neg = M * torch.exp(S * pos_neg_weight)
    pos = torch.cat([torch.diag(S, node_num), torch.diag(S, -node_num)], dim=0)
    pos = torch.exp(pos * pos_weight)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
    return infoNEC

# 原版
# class ClusterConsistencyLoss(nn.Module):
#     def __init__(self, n_clusters, in_channels, temperature=0.5, fuse_mode:typing.Literal['attention', 'cat', 'add']='attention'):
#         super(ClusterConsistencyLoss, self).__init__()
#         self.n_clusters = n_clusters
#         self.temperature = temperature
#         self.attention = Attention(in_channels)
#         self.fuse_mode = fuse_mode
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.ReLU(),
#             nn.Linear(in_channels, n_clusters),
#             nn.Softmax(dim=1)
#         )
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, z1: torch.Tensor, z2: torch.Tensor):
#         if self.fuse_mode == 'attention':
#             zf, att = self.attention(torch.stack([z1, z2], dim=1))  # 融合特征
#         elif self.fuse_mode == 'cat':    # 会出错
#             zf = torch.cat((z1, z2), dim=1)
#         elif self.fuse_mode == 'add':
#             zf = (z1 + z2) / 2
#
#         predict_labels, centers = kmeans_clustering_torch(zf, self.n_clusters)
#
#         prototypes = self.compute_centers(zf, predict_labels.cpu().numpy())
#
#         # 分别求centers与z1,z2各个元素的相似度
#         sim1 = torch.zeros(z1.size(0), self.n_clusters).to(z1.device)
#         sim2 = torch.zeros(z2.size(0), self.n_clusters).to(z1.device)
#         for i in range(self.n_clusters):
#             sim1[:, i] = torch.cosine_similarity(z1, prototypes[i].view(1, -1), dim=1)
#             sim2[:, i] = torch.cosine_similarity(z2, prototypes[i].view(1, -1), dim=1)
#
#
#         soft_label1 = distributed_sinkhorn(sim1)
#         soft_label2 = distributed_sinkhorn(sim2)
#
#         predict_labels_one_hot = torch.zeros(zf.size(0), self.n_clusters).to(zf.device)
#         predict_labels_one_hot.scatter_(1, predict_labels.view(-1, 1), 1)
#         clu_loss = self.criterion(soft_label1, predict_labels_one_hot) + self.criterion(soft_label2, predict_labels_one_hot)
#         return clu_loss, soft_label1, soft_label2, zf, predict_labels, prototypes
#
#     def compute_centers(self, x, cluster_labels):
#         n_samples = x.size(0)
#         if len(torch.from_numpy(cluster_labels).size()) > 1:
#             weight = cluster_labels.T
#         else:
#             weight = torch.zeros(self.n_clusters, n_samples).to(x)
#             weight[cluster_labels, torch.arange(n_samples)] = 1
#         weight = F.normalize(weight, p=1, dim=1)
#         centers = torch.mm(weight, x)
#         centers = F.normalize(centers, dim=1)
#         return centers

# 可導，精度很差
# class ClusterConsistencyLoss(nn.Module):
#     def __init__(
#         self,
#         n_clusters: int,
#         in_channels: int,
#         temperature: float = 0.5,
#         fuse_mode: typing.Literal['attention', 'cat', 'add'] = 'attention'
#     ):
#         super(ClusterConsistencyLoss, self).__init__()
#         self.n_clusters = n_clusters
#         self.temperature = temperature
#         self.fuse_mode = fuse_mode
#
#         # 融合模块
#         if fuse_mode == 'attention':
#             self.attention = Attention(in_channels)
#
#         # 可学习的分类头用于 soft cluster assignment
#         self.cluster_head = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.ReLU(),
#             nn.Linear(in_channels, n_clusters),
#             nn.Softmax(dim=1)
#         )
#
#         # 损失函数
#         # self.criterion = nn.KLDivLoss(reduction='batchmean')
#         self.criterion = nn.CrossEntropyLoss()
#
#     def compute_prototypes(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#         """
#         根据 soft 权重计算 prototype 向量
#         :param x: 特征张量 [B, D]
#         :param weights: soft 分布 [B, K]
#         :return: prototypes [K, D]
#         """
#         weights = F.normalize(weights, p=1, dim=0)  # 按列归一化为概率分布
#         prototypes = torch.mm(weights.t(), x)  # 加权平均
#         prototypes = F.normalize(prototypes, dim=1)  # L2 归一化
#         return prototypes
#
#     def forward(self, z1: torch.Tensor, z2: torch.Tensor):
#         # 融合特征
#         if self.fuse_mode == 'attention':
#             zf, _ = self.attention(torch.stack([z1, z2], dim=1))
#         elif self.fuse_mode == 'cat':   # 會出錯
#             zf = torch.cat((z1, z2), dim=1)
#         elif self.fuse_mode == 'add':
#             zf = (z1 + z2) / 2
#
#         # 使用可学习 head 预测 soft cluster 分布
#         with torch.no_grad():
#             q_zf = self.cluster_head(zf)
#         p_zf = q_zf.detach()
#
#         # 对 z1 和 z2 分别预测 cluster 分布
#         q1 = self.cluster_head(z1)
#         q2 = self.cluster_head(z2)
#
#         # Sinkhorn 分布或其他分布归一化
#         p1 = distributed_sinkhorn(q1)
#         p2 = distributed_sinkhorn(q2)
#
#         predict_labels = torch.argmax(q_zf, dim=1)  # 不可導
#         # 获取 prototypes（可导）
#         prototypes = self.compute_prototypes(zf, q_zf)
#
#         predict_labels_one_hot = torch.zeros(zf.size(0), self.n_clusters).to(zf.device)
#         predict_labels_one_hot.scatter_(1, predict_labels.view(-1, 1), 1)
#         loss = self.criterion(q1, predict_labels_one_hot) + self.criterion(q2, predict_labels_one_hot)
#         # # 计算一致性损失（KL 散度）
#         # loss = (
#         #     self.criterion(F.log_softmax(q1 / self.temperature, dim=1), p_zf)
#         #     + self.criterion(F.log_softmax(q2 / self.temperature, dim=1), p_zf)
#         # ) / 2
#
#         return loss, p1, p2, zf, predict_labels, prototypes

class ClusterConsistencyLoss(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        in_channels: int,
        temperature: float = 0.5,
        fuse_mode: Literal['attention', 'cat', 'add'] = 'attention'
    ):
        super(ClusterConsistencyLoss, self).__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.fuse_mode = fuse_mode

        # 融合模块
        if fuse_mode == 'attention':
            self.attention = Attention(in_channels)

        # 可学习的分类头用于 soft cluster assignment
        self.cluster_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_clusters),
            nn.Softmax(dim=1)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 融合特征
        if self.fuse_mode == 'attention':
            zf, _ = self.attention(torch.stack([z1, z2], dim=1))
        elif self.fuse_mode == 'cat':
            zf = torch.cat((z1, z2), dim=1)
        elif self.fuse_mode == 'add':
            zf = (z1 + z2) / 2

        # 获取 soft cluster 分布
        q_zf = self.cluster_head(zf)
        with torch.no_grad():
            p_zf = q_zf.detach().clone()

        # 获取 prototypes（soft 权重）
        prototypes = self.compute_prototypes(zf, q_zf)

        # 计算 z1, z2 到 prototypes 的相似度（模拟原来的 sim1, sim2）
        sim1 = self.cos_sim_to_prototypes(z1, prototypes)
        sim2 = self.cos_sim_to_prototypes(z2, prototypes)

        # Sinkhorn 归一化（或直接 softmax）
        soft_label1 = distributed_sinkhorn(sim1)
        soft_label2 = distributed_sinkhorn(sim2)

        # 构造 pseudo-labels（hard label，来自当前 cluster_head 输出）
        # predict_labels = torch.argmax(q_zf, dim=1).detach()  # 用于记录/分析
        predict_labels, centers = kmeans_clustering_torch(zf, self.n_clusters)
        # 将 hard label 转成 one-hot 分布
        predict_labels_one_hot = F.one_hot(predict_labels, num_classes=self.n_clusters).float().to(
            predict_labels.device)

        # 然后使用 KL 散度
        loss = self.criterion(F.log_softmax(sim1 / self.temperature, dim=1), predict_labels_one_hot)

        return loss, soft_label1, soft_label2, zf, predict_labels, prototypes

    def compute_prototypes(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        根据 soft 权重计算 prototype 向量
        :param x: [B, D]
        :param weights: [B, K]
        :return: [K, D]
        """
        weights = F.softmax(weights / self.temperature, dim=0)
        prototypes = torch.mm(weights.t(), x)       # 加权平均
        prototypes = F.normalize(prototypes, dim=1) # L2 归一化
        return prototypes

    def cos_sim_to_prototypes(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算 x 中每个样本与所有 prototypes 的余弦相似度
        :param x: [B, D]
        :param prototypes: [K, D]
        :return: [B, K]
        """
        x_normalized = F.normalize(x, dim=1)
        p_normalized = F.normalize(prototypes, dim=1)
        return torch.mm(x_normalized, p_normalized.t())


class HardSampleAwareLoss_Pre(nn.Module):
    def __init__(self, n_samples, update_mode: typing.Literal['reselect', 'progressive']='reselect', beta=3, device='cpu'):
        '''
        ## Hard Sample Loss
        ---
        n_samples: 超像素样本数量
        update_mode: hard sample更新模式 reselect: 每轮重选 progressive: 在上轮基础上选
        '''
        super(HardSampleAwareLoss_Pre, self).__init__()
        self.hard_sample_idx = None
        self.reliable_sample_idx = None
        self.update_mode = update_mode
        self.n_samples = n_samples
        self.beta = beta
        # positive and negative sample pair index matrix
        self.mask = torch.ones([n_samples * 2, n_samples * 2]) - torch.eye(n_samples * 2)
        self.mask.to(device)

        self.act = nn.Tanh()
        self.pos_weight = torch.ones(n_samples * 2).to(device)
        self.pos_neg_weight = torch.ones([n_samples * 2, n_samples * 2]).to(device)
        self.tmp_pos_weight = self.pos_weight.clone().detach()
        self.tmp_pos_neg_weight = self.pos_neg_weight.clone().detach()

    def forward(self, z1, z2, soft_label1, soft_label2, predict_labels, centers, beta):    # 这里的predict_labels是zf聚类的结果

        self.beta = beta

        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        self.pos_weight.data = self.tmp_pos_weight.data
        self.pos_neg_weight.data = self.tmp_pos_neg_weight.data

        label1 = torch.argmax(soft_label1, dim=1)
        label2 = torch.argmax(soft_label2, dim=1)
        # 取两个label不同处的下标
        hard_sample_idx = torch.where(label1 != label2)
        # 取两个label相同处的下标
        reliable_sample_idx = torch.where(label1 == label2)

        if self.hard_sample_idx is None:
            self.hard_sample_idx = hard_sample_idx
        else:
            if self.update_mode == 'reselect':
                self.hard_sample_idx = hard_sample_idx
            elif self.update_mode == 'progressive':
                # 取 hard_sample_idx 和 self.hard_sample_idx 的交集
                self.hard_sample_idx = torch.intersect(self.hard_sample_idx, hard_sample_idx)

        if self.reliable_sample_idx is None:
            self.reliable_sample_idx = reliable_sample_idx
        else:
            if self.update_mode == 'reselect':
                self.reliable_sample_idx = reliable_sample_idx
            elif self.update_mode == 'progressive':
                # 取 reliable_sample_idx 和 self.reliable_sample_idx 的交集
                self.reliable_sample_idx = torch.intersect(self.reliable_sample_idx, reliable_sample_idx)



        S = comprehensive_similarity(z1, z2)
        # S = S / S.max()     # 不这样会求出nan
        # S = self.act(S)
        H = torch.cat([self.reliable_sample_idx[0], self.reliable_sample_idx[0] + self.n_samples], dim=0)
        H_mat = np.ix_(H.cpu(), H.cpu())
        # calculate new weight of sample pair
        M, M_mat = pseudo_matrix(predict_labels, S, self.n_samples, self.beta)
        # calculate hard sample aware contrastive loss
        loss = hard_sample_aware_infoNCE(S, self.mask, self.pos_neg_weight, self.pos_weight, self.n_samples)

        # update weight
        self.tmp_pos_weight[H].data = M[H].data
        self.tmp_pos_neg_weight[H_mat].data = M_mat[H_mat].data

        return loss, label1, label2


