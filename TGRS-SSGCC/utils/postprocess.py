import torch
import numpy as np
from scipy.sparse.linalg import svds
from torch_clustering import PyTorchKMeans
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from utils.evaluation import best_match


def label_to_onehot(labels):
    """
    将标签转换成onehot编码
    :param labels: 标签列表
    :return: onehot编码矩阵
    """
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_samples = len(labels)
    onehot = np.zeros((n_samples, n_labels))
    for i, label in enumerate(labels):
        onehot[i, np.where(unique_labels == label)[0][0]] = 1
    return onehot


def kmeans_clustering_torch(features, n_cluster):
    """
    ## 对点进行KMeans聚类,返回每个点属于每个聚类的概率
    ---
    :param points: 大小为N*S的点的特征矩阵,N为点的数量,S为每个点的特征维度
    :param n_clusters: 聚类的数量
    :return: 大小为N*K的矩阵,N为点的数量,K为聚类的数量,代表每个点为任何一种类别的概率
    """
    kmeans = PyTorchKMeans(n_clusters=n_cluster, init='k-means++', distributed=True, verbose=False)
    labels = kmeans.fit_predict(features.detach())
    centers = kmeans.cluster_centers_
    return labels, centers

# def kmeans_clustering(features, n_clusters):
#     """
#     ## 对点进行KMeans聚类,返回每个点属于每个聚类的概率
#     ---
#     :param points: 大小为N*S的点的特征矩阵,N为点的数量,S为每个点的特征维度
#     :param n_clusters: 聚类的数量
#     :return: 大小为N*K的矩阵,N为点的数量,K为聚类的数量,代表每个点为任何一种类别的概率
#     """
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(features)
#     centers = kmeans.cluster_centers_
#     distances = kmeans.transform(features)
#     exp_distances = np.exp(-distances ** 2 / 2) + 1e-7  # 计算距离的指数函数
#     probs = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)  # 计算每个点属于每个聚类的概率
#     pred = np.argmax(probs, axis=1)
#     return probs, pred, centers


def spectral_clustering(affinity_mat, n_clusters):
    """
    ## 对点进行谱聚类,返回每个点属于每个聚类的概率
    ---
    :param points: 大小为N*S的点的特征矩阵,N为点的数量,S为每个点的特征维度
    :param n_clusters: 聚类的数量
    :return: 大小为N*K的矩阵,N为点的数量,K为聚类的数量,代表每个点为任何一种类别的概率
    """
    spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed',
                                  assign_labels='discretize', random_state=42)
    spectral.fit(affinity_mat)
    labels = spectral.labels_
    n_samples = affinity_mat.shape[0]
    probs = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        probs[i, labels[i]] = 1
    pred = np.argmax(probs, axis=1)
    return probs, pred


# def sp_px_cluster(affinity_mat, px_features, px_idx, n_clusters):
#     num_sp, num_px = px_idx.shape
#     px_idx = px_idx.reshape(-1)
#     px_features = px_features[px_idx].reshape(num_sp, num_px, -1)
#     px_to_sp_probs, px_to_sp_pred = kmeans_clustering(px_features, n_clusters)     # 由像素算出的超像素标签概率
#     sp_pred = affinity_to_pixellabels(affinity_mat.detach().cpu().numpy(), n_clusters)
#     sp_pred = best_match(px_to_sp_pred.cpu().numpy(), sp_pred)
#     sp_probs = label_to_onehot(sp_pred)

#     return px_to_sp_probs, px_to_sp_pred, torch.from_numpy(sp_probs).float(), torch.from_numpy(sp_pred).float()


def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = np.matmul(association_mat, sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')


def affinity_to_pixellabels(affinity_mat, n_clusters):
    # Coef = thrC(affinity_mat, 0.8)
    # y_pre, C = post_proC(Coef, n_clusters, 8, 18)

    affinity_mat = 0.5 * (np.abs(affinity_mat) + np.abs(affinity_mat.T))
    # import matplotlib.pyplot as plt
    # print(affinity_mat.diagonal().sum())
    # plt.imshow(affinity_mat, cmap='hot')
    # plt.show()

    # # using pure SC
    spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize', random_state=42)
    spectral.fit(affinity_mat)
    y_pre = spectral.fit_predict(affinity_mat)
    # kmeans = cluster.KMeans(n_clusters=n_clusters, max_iter=500, random_state=42)
    # y_pre = kmeans.fit_predict(affinity_mat) + 1

    y_pre = y_pre.astype('int')
    return y_pre


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L