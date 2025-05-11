'''
# Contrastive Loss
'''

import torch
import torch.nn as nn

class InstanceLossSP(nn.Module):      
    '''
    ## 超像素对比loss
    '''
    def __init__(self, temperature=0.5, n_neighbor=10):
        super(InstanceLossSP, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.n_neighbor = n_neighbor

    def forward(self, z, aff_mat=None):
        # # 医学文章的写法
        # n_samples, dim_fea = z.size()
        # mask = torch.ones((n_samples, n_samples))
        # mask = mask.fill_diagonal_(0).bool().to(z.device)   # 对角填0 自己不是自己的负样本
        # if aff_mat is None:
        #     z = z / z.norm(dim=1)[:, None]
        #     sim = torch.exp(torch.matmul(z, z.T) / self.temperature)
        # else:
        #     sim = torch.exp(aff_mat / self.temperature)
        # e_all = sim[mask].reshape(n_samples, -1).sum(1)
        # e_sim = torch.diag(sim)
        # loss = -torch.log(e_sim / e_all)
        # loss = loss.mean()
        # return loss
    
        ## +-
        n_samples, dim_fea = z.size()
        mask = torch.ones((n_samples, n_samples))
        mask = mask.fill_diagonal_(0).bool().to(z.device)
        z = z / z.norm(dim=1)[:, None]
        similarity_matrix = torch.exp(torch.mm(z, z.T) / self.temperature)
        e_all = similarity_matrix[mask].reshape(n_samples, -1).sum(1)
        k_nearest_indices = torch.topk(similarity_matrix, k=self.n_neighbor+1, dim=1, largest=True, sorted=True).indices[:, 1:]
        # 正样本相似度
        e_sim = torch.gather(similarity_matrix, dim=1, index=k_nearest_indices).sum(1)
        loss = -torch.log(e_sim / e_all)    # 为了使loss变小，需要e_sim变大，e_all变小，即最大化正样本相似度，最小化其余样本相似度
        loss = loss.mean()
        return loss

        ## NCL 的写法
        # n_samples, dim_fea = z.size()
        # mask = torch.ones((n_samples, n_samples))
        # mask = mask.fill_diagonal_(0).bool().to(z.device)   # 对角填0 自己不是自己的负样本
        # similarity_matrix = torch.mm(z, z.T) / self.temperature
        # similarity_matrix = similarity_matrix[mask].view((n_samples, -1))
        # _, pos_indx = torch.topk(similarity_matrix, k=self.n_neighbor, dim=1, largest=True, sorted=True)  # 每个节点选取k个最近邻

        # pos_sim = torch.gather(similarity_matrix, dim=1, index=pos_indx)  # 根据最近邻的索引选取相似度
        # loss = (- torch.log(pos_sim / similarity_matrix.sum(dim=-1).view(n_samples, -1) + 1e-8)).sum(dim=1).mean()  # 计算损失
        # return loss

        ## propos的写法（正负样本）
        # n_samples, dim_fea = z.size()
        # mask = torch.ones((n_samples, n_samples))
        # mask = mask.fill_diagonal_(0).bool().to(z.device)   # 对角填0 自己不是自己的负样本

        # if aff_mat is None:
        #     z = z / z.norm(dim=1)[:, None]
        #     similarity_matrix = torch.mm(z, z.T) / self.temperature
        #     k_nearest_indices = torch.topk(similarity_matrix, k=self.n_neighbor+1, dim=1, largest=True, sorted=True).indices[:, 1:]

        #     # 取正样本
        #     positive_samples = torch.gather(similarity_matrix, dim=1, index=k_nearest_indices)

        #     # 取负样本
        #     mask = torch.ones(n_samples, n_samples, dtype=torch.bool)
        #     mask.scatter_(1, k_nearest_indices, 0)
        #     negative_samples = torch.masked_select(similarity_matrix, mask).reshape(n_samples, -1)


        # else:
        #     negative_samples = torch.exp(aff_mat[mask].reshape(n_samples, -1) / self.temperature)
        # labels = torch.zeros(n_samples).long().to(z.device)
        # # 将正样本和负样本拼接在一起
        # logits = torch.cat((positive_samples, negative_samples), dim=1)
        # loss = self.criterion(logits, labels)
        # return loss
    
        ## propos的写法（仅负样本）
        # n_samples, dim_fea = z.size()
        # mask = torch.ones((n_samples, n_samples))
        # mask = mask.fill_diagonal_(0).bool().to(z.device)   # 对角填0 自己不是自己的负样本

        # if aff_mat is None:
        #     # z = z / z.norm(dim=1)[:, None]
        #     sim = torch.mm(z, z.T) / self.temperature
        #     negative_samples = sim[mask].reshape(n_samples, -1)
        # else:
        #     negative_samples = torch.exp(aff_mat[mask].reshape(n_samples, -1) / self.temperature)
        # labels = torch.zeros(n_samples).long().to(z.device)
        # loss = self.criterion(negative_samples, labels)
        # return loss

class InstanceLossPX(nn.Module):      
    '''
    ## 像素对比loss
    '''
    def __init__(self, temperature=0.5):
        '''
        '''
        super(InstanceLossPX, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')       # 超像素内所有像素对比loss的和看成一个超像素的loss

    def forward(self, z, z_px, asso_mat):
        '''
        z: 超像素特征 n_sps * S
        z_px: 像素特性 n_pxs * S 或 n_sps * n_px_per_so * S
        '''

        try:
            n_samples, n_px_per_sp, dim_fea = z_px.size()
            mask = torch.ones((n_px_per_sp, n_px_per_sp))
            mask = mask.fill_diagonal_(0).bool().to(z_px.device)   # 对角填0 自己不看作自己的正样本
            loss = 0
            for i in range(n_samples):          # 每个超像素内的所有像素互相间都是正例
                z = z_px[i]
                sim = torch.mm(z, z.T) / self.temperature
                positive_samples = sim[mask].reshape(n_px_per_sp, -1)
                labels = torch.zeros(n_px_per_sp).long().to(z.device)
                loss += self.criterion(positive_samples, labels)
            return loss / n_samples
        except:
            ## 超像素内所有像素都是正样本
            z = z / z.norm(dim=1)[:, None]
            affinity_matrix = torch.exp(torch.matmul(z, z.T) / self.temperature)
            affinity_matrix_sum = torch.sum(affinity_matrix, 0)

            loss = 0
            n_pxs, n_sps = asso_mat.size()
            for i in range(n_sps):
                mask = asso_mat[:, i].bool()
                pixels_feature = z_px[asso_mat[:, i].bool(), :]
                pixels_feature_norm = pixels_feature / pixels_feature.norm(dim=1)[:, None]
                affinity_matrix_pixels = torch.mm(z[i:i+1], pixels_feature_norm.t())
                affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / self.temperature)
                e_sim = affinity_matrix_pixels.mean()
                e_dis = affinity_matrix_sum[i] - affinity_matrix[i,i]
                loss += -torch.log(e_sim / (e_sim + e_dis))
        return loss / n_sps

        
    
class InstanceLoss(nn.Module):
    '''
    
    '''
    def __init__(self, batch_size, temperature):
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
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # 正样本，也就是增强后的和原来的对比 a & a', b & b'
        negative_samples = sim[self.mask].reshape(N, -1)                        # 负样本，也就是 a/a' & (b, b', c, c'), b/b' & (a, a', c, c')

        # 将positive_samples和negative_samples对应的标签设为0，自然的，对应的positive_samples就是正例，negative_samples就是负例
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss