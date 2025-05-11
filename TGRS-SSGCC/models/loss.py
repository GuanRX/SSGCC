import torch
import torch.nn as nn
# from sklearn.neighbors import kneighbors_graph
import numpy as np
# from utils.postprocess import affinity_to_pixellabels
from torch_clustering import PyTorchKMeans
from utils.postprocess import sp_px_cluster, kmeans_clustering_torch, kmeans_clustering
from utils.evaluation import get_y_preds
from models.ct_loss import *


# class ProtoLoss(nn.Module):
#     def __init__(self, n_clusters):
#         """
#         ## 原型损失
#         """
#         super(ProtoLoss, self).__init__()
#         self.n_clusters = n_clusters
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         kwargs = {
#             'distributed': True,
#             'random_state': 0,
#             'n_clusters': self.n_clusters,
#             'verbose': False
#         }
#         clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

#         psedo_labels = clustering_model.fit_predict(x.detach())
#         cluster_centers = clustering_model.cluster_centers_
#         centers = cluster_centers[psedo_labels]
#         return self.criterion(x, centers) / self.n_clusters


class PxToSpLoss(nn.Module):
    '''
    ## 像素-超像素损失
    '''
    def __init__(self, n_clusters) -> None:
        super(PxToSpLoss, self).__init__()
        self.n_clusters = n_clusters
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, affinity_mat, px_features, pixel_choosed_idx):
        px_to_sp_probs, px_to_sp_pred, sp_probs, sp_pred = sp_px_cluster(affinity_mat, px_features, pixel_choosed_idx, self.n_clusters)
        return self.criterion(px_to_sp_probs, sp_probs)


class InstanceLoss(nn.Module):      
    '''
    ## 正负例对比loss
    '''
    def __init__(self, batch_size, temperature=0.5):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        mask = torch.ones((self.batch_size, self.batch_size))  # 构造全1矩阵
        mask = mask.fill_diagonal_(0).bool().to(z_i.device)  # 将对角线的1设置为0，转成bool型，移到指定设备上
        z_i, z_j = z_i[mask].view((self.batch_size, -1)), z_j[mask].view((self.batch_size, -1)).to(z_i.device)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class NCL_Loss_Graph(nn.Module):
    """
    neighborhood contrastive learning loss based on using a N*N affinity/similarity matrix/adjacent matrix
    """
    def __init__(self, n_neighbor=5, temperature=1.):
        """
        :param n_neighbor: 选取相邻的n个节点进行计算loss
        :param temperature: softmax的温度参数
        """
        super(NCL_Loss_Graph, self).__init__()
        self.n_neighbor = n_neighbor
        self.temperature = temperature
        self.BCE = nn.BCELoss()

    def forward(self, affinity_pred, affinity_init):
        # n_clusters = 8
        # labels = affinity_to_pixellabels(affinity_pred.detach().cpu().numpy(), n_clusters)
        # # 初始化正样本索引矩阵
        # pos_samples_indx = np.zeros((labels.shape[0], self.n_neighbor), dtype=np.int64)

        # # 对于每个样本，找出和它属于同一类的n个样本
        # # for i in range(self.n_neighbor):
        # for i in range(labels.shape[0]):
        #     pos_samples_mask = labels == labels[i]
        #     pos_samples_mask[i] = False
        #     pos_samples_mask = np.argwhere(pos_samples_mask==True).flatten()
        #     pos_samples_mask[pos_samples_mask > i] -= 1
        #     pos_samples_indx[i] = np.random.choice(pos_samples_mask, self.n_neighbor, replace=True)
            # if labels[i] == i:
            #     # 找到和样本j属于同一类的其他样本
            #     pos_samples_mask = (labels == i) & (labels != labels[j])
            #     pos_samples_indx[j][i] = np.random.choice(np.flatnonzero(pos_samples_mask), size=1)[0]
        ...

        # affinity_init = kneighbors_graph(z.detach().cpu().numpy(), n_neighbors=50, mode='distance', include_self=False).toarray()
        # affinity_init = torch.from_numpy(affinity_init).to(affinity_pred.device)

        N = affinity_init.shape[0]
        mask = torch.ones((N, N))  # 构造全1矩阵
        mask = mask.fill_diagonal_(0).bool().to(affinity_pred.device)  # 将对角线的1设置为0，转成bool型，移到指定设备上
        affinity_pred, affinity_init = affinity_pred[mask].view((N, -1)), affinity_init[mask].view((N, -1)).to(affinity_pred.device)

        # return self.ins_loss.forward(affinity_pred, affinity_init)
        _, pos_indx = torch.topk(affinity_init, k=self.n_neighbor, dim=1, largest=True, sorted=True)  # 每个节点选取k个最近邻
        sim_matrix = torch.exp(affinity_pred / self.temperature)  # 计算相似度矩阵
        # pos_indx = torch.from_numpy(affinity_init).to(affinity_pred.device)
        pos_sim = torch.gather(sim_matrix, dim=1, index=pos_indx)  # 根据最近邻的索引选取相似度
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1).view(N, -1) + 1e-8)).sum(dim=1).mean()  # 计算损失
        return loss
    
    
class GraphRegularizer(nn.Module):

    def __init__(self, aff_init, n_neighbor=10):
        super(GraphRegularizer, self).__init__()

        res, pos_indx = torch.topk(aff_init, k=n_neighbor, dim=1, largest=True, sorted=True)
        A = torch.zeros((aff_init.shape[0], aff_init.shape[0]))
        A = A.scatter(src=torch.ones_like(res), dim=1, index=pos_indx)
        A = (A + A.t())/2
        # A = torch.where(A>=2, A, torch.zeros_like(A))/2
        # A.fill_diagonal_(0)
        D = torch.pow(torch.sum(A, dim=1), -0.5)
        D = torch.diag(D)
        L = torch.eye(A.shape[0]) - torch.matmul(torch.matmul(D, A), D)
        self.L = L

    def forward(self, pred_aff):
        L = self.L.to(pred_aff.device)
        loss = torch.trace(pred_aff.t().mm(L).mm(pred_aff))
        return loss


# class Loss(nn.Module):
#     def __init__(self, n_clusters, px_idx, trade_off=0.005, trade_off_sparse=0.005, n_neighbor=10, temperature=0.1, regularizer='NC', aff_init=None):
#         """
#         :param n_clusters:
#         :param px_idx:
#         :param trade_off:
#         :param trade_off_sparse:
#         :param n_neighbor:
#         :param temperature:
#         :param regularizer:  'NC': neighbor contrastive; 'L1'/'L2', None
#         :param aff_init:
#         """
#         super(Loss, self).__init__()
#         self.regularizer = regularizer
#         self.trade_off = trade_off
#         self.trade_off_sparse = trade_off_sparse
#         self.criterion_mse_recon = nn.MSELoss()
#         self.criterion_mse_latent = nn.MSELoss()
#         self.aff_init = aff_init
#         self.px_idx = px_idx        # 选择的像素编号

#         self.eta = nn.Parameter(torch.Tensor([0., -1, -10.]), requires_grad=True)

#         # self.eta = nn.Parameter(torch.Tensor([0.1, -1., -10.]), requires_grad=True)
#         if aff_init is not None and not isinstance(aff_init, torch.Tensor):
#             self.aff_init = torch.from_numpy(aff_init).float()
#             self.graph_reg = GraphRegularizer(self.aff_init, n_neighbor=n_neighbor)
#         self.ncl_criterion = NCL_Loss_Graph(n_neighbor=n_neighbor, temperature=temperature)

#         self.const_criterion = InstanceLoss(aff_init.shape[0], temperature) # 对比loss
#         self.proto_criterion = ProtoLoss(n_clusters)                        # 原型loss
#         self.px_sp_criterion = PxToSpLoss(n_clusters)                       # 像素超像素loss

#     def forward(self, model, x_true, x_pred, z, z_pred):
#         loss_regularization = torch.tensor(0., requires_grad=False).to(x_true.device)
#         for name, param in model.named_parameters():
#             if 'affinity_mat' in name:
#                 # loss_regularization += torch.sum(torch.norm(param, p=2))
#                 # loss_regularization += torch.abs(torch.diagonal(param)).sum()
#                 # loss_regularization += self.ncl_criterion(param, self.aff_init)
#                 if self.regularizer == 'NC':
#                     affinity_pred = param
#                     loss_regularization += self.ncl_criterion(param, self.aff_init)
#                     # loss_regularization += self.ncl_criterion(param, self.aff_init)
#                 elif self.regularizer == 'L1':
#                     loss_regularization += torch.sum(torch.abs(param))
#                 elif self.regularizer == 'L2':
#                     loss_regularization += torch.norm(param, p='fro')
#                 elif self.regularizer == 'None':
#                     loss_regularization = loss_regularization
#                 elif self.regularizer == 'Graph':
#                     loss_regularization += self.graph_reg(param)
#                 else:
#                     raise Exception('invalid regularizer!')
#         loss_model_recon = self.criterion_mse_recon(x_pred, x_true)
#         loss_se_recon = self.criterion_mse_latent(z_pred, z)
#         if self.regularizer == 'NC':
#             loss_ = torch.stack([loss_model_recon, loss_se_recon, loss_regularization]).to(x_true.device)
#             loss = (loss_ * torch.exp(-self.eta) + self.eta).sum()
#         else:
#             loss = loss_model_recon + self.trade_off * loss_se_recon + self.trade_off_sparse * loss_regularization

#         # loss_proto = self.proto_criterion(z_pred)
#         # loss += loss_proto        # 加原型损失

#         # loss_const = self.const_criterion(affinity_pred, self.aff_init)
#         # loss += loss_const        # 加对比损失

#         # loss_px_sp = self.px_sp_criterion(affinity_pred, x_pred, self.px_idx)
#         # loss += loss_px_sp        # 加像素超像素损失

#         return loss, loss_model_recon, loss_se_recon, loss_regularization, torch.exp(-self.eta)



class Loss(nn.Module):
    def __init__(self, n_clusters, px_idx, temperature=0.5, beta1=1, beta2=1):
        """
        :param n_clusters:
        :param px_idx:
        :param trade_off:
        :param trade_off_sparse:
        :param n_neighbor:
        :param temperature:
        :param regularizer:  'NC': neighbor contrastive; 'L1'/'L2', None
        :param aff_init:
        """
        super(Loss, self).__init__()
        self.px_idx = px_idx        # 选择的像素编号
        self.n_clusters = n_clusters
        self.criterion_mse_latent = nn.MSELoss()
        self.posi_criterion = InstanceLossPX(temperature)                   # 正样本是超像素内的所有像素
        self.nega_criterion = InstanceLossSP(temperature)                   # 负样本是所有超像素
        self.px_sp_criterion = nn.CrossEntropyLoss()
        self.eta = nn.Parameter(torch.Tensor([0., -1., 0.]), requires_grad=True)
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, x_pred, z, asso_mat=None):
        # loss_se_recon = self.criterion_mse_latent(z_pred, z)              # 自表达重构损失

        num_sp, num_px = self.px_idx.shape
        px_idx = self.px_idx.reshape(-1)
        px_features = x_pred[px_idx].reshape(num_sp, num_px, -1)
        loss_positive = self.posi_criterion(z, x_pred, asso_mat)            # 像素对比损失
        loss_negative = self.nega_criterion(z)                              # 超像素对比损失

        px_to_sp_probs, px_to_sp_pred = kmeans_clustering_torch(px_features.detach(), self.n_clusters)     # 由像素算出的超像素标签概率
        sp_probs, sp_pred = kmeans_clustering(z.detach().cpu().numpy(), self.n_clusters)
        sp_pred = get_y_preds(px_to_sp_pred.cpu().numpy(), sp_pred, self.n_clusters)
        sp_probs = torch.from_numpy(sp_probs).float()
        sp_pred = torch.from_numpy(sp_pred).float()

        unclean_idx = px_to_sp_pred != sp_pred
        print(f'num unclean samples = [{unclean_idx.sum().item()}/{num_sp}]')
        loss_px_sp = self.px_sp_criterion(px_to_sp_probs[unclean_idx, :], sp_probs[unclean_idx, :])
        # loss_px_sp = self.px_sp_criterion(px_to_sp_probs, sp_probs)

        loss = loss_positive + loss_negative + loss_px_sp
        # loss = loss_positive + loss_px_sp
        # loss_ = torch.stack([loss_positive, loss_negative, loss_px_sp]).to(x_pred.device)
        # loss = (loss_ * torch.exp(-self.eta) + self.eta).sum()
        return loss, px_to_sp_pred, sp_pred, px_features#, torch.exp(-self.eta)

