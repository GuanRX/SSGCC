# sp: 邻域特征 使用GAE提取特征    px: 全图直接卷积 使用AE提取特征
import sys
import json
import os
import torch
import time
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import minmax_scale
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn

from models.dualspnet import DualSPNet
from models.losses import ClusterConsistencyLoss, HardSampleAwareLoss_Pre
from dataset import HSIDataset
from utils.postprocess import spixel_to_pixel_labels, affinity_to_pixellabels
from utils.evaluation import cluster_accuracy, get_parameter_number, purity_score
from utils.tsne import *

def get_config(config_path):
    f = open(config_path, 'r')
    return json.load(f)


def theoretical_max_acc(seg, gt):# 最高精度参考
    res = np.zeros_like(gt)
    for num in np.unique(seg):
        mask = seg == num
        label = gt[mask]
        c = Counter(label).most_common()         # 出现频率最多的标签
        if c[0][0] == 0 and len(c) > 1:
            res[mask] = c[1][0]                 # 0是背景,计算精度会忽略
        else:
            res[mask] = c[0][0]
    indx = np.where(gt != 0)
    y_true = gt[indx]
    y_best = res[indx]
    acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_true, y_best, return_aligned=False)
    print('\n[ theoretical max acc ]:')
    print_table([[acc, kappa, nmi, ari, pur]], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])

def evaluate_affinity(model, dataset):
    model.eval()  # # training flag
    for name, param in model.named_parameters():
        if 'affinity_mat' in name:
            affinity_mat_ = param.detach().cpu().numpy()
    y_pre_sp = affinity_to_pixellabels(affinity_mat_, dataset.n_classes)
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, dataset.association_mat.cpu().numpy())
    class_map = y_pre_pixel.reshape(dataset.gt.shape)

    indx = np.where(dataset.gt != 0)
    y_target = dataset.gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)

def evaluate(y_pre_sp, association_mat, gt):
    if isinstance(y_pre_sp, torch.Tensor):
        y_pre_sp = y_pre_sp.cpu().numpy()
    if isinstance(association_mat, torch.Tensor):
        association_mat = association_mat.cpu().numpy()
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, association_mat)
    class_map = y_pre_pixel.reshape(gt.shape)

    indx = np.where(gt != 0)
    y_target = gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)


def print_table(data, headers, col_width=12):
    num_cols = len(headers)
    col_spacing = 3
    
    # create row format
    row_format = "".join(["{:>"+str(col_width)+"}"]*(num_cols))
    row_separator = "".join(["{:->"+str(col_width)+"}"]*(num_cols))
    
    # print header
    print(row_format.format(*headers))
    print(row_separator.format(*["" for _ in range(num_cols)]))
    
    # print data rows
    for i, row in enumerate(data):
        row_values = [f"{val:.4f}" if isinstance(val, float) else val for val in row]
        print(row_format.format(*row_values))


        
def report_csv(curve, dataset_name):
    index = ['Epoch ' + str(i + 1) for i in range(len(curve) - 1)] + ['max']
    columns = ['acc', 'kappa', 'nmi', 'ari', 'pur', 'loss',
               'acc_std', 'kappa_std', 'nmi_std', 'ari_std', 'pur_std', 'loss_std']
    dt = datetime.fromtimestamp(time.time())
    df = pd.DataFrame(curve, index=index, columns=columns)
    save_path = 'result/' + dataset_name + '_{}.csv'.format(dt.strftime('%Y_%m_%d_%H_%M_%S'))
    df.to_csv(save_path)

def main(config, trained_model=None):

    device = torch.device(config['device'])
    dataset_name = config['dataset_name']

    gae_n_enc_1, gae_n_enc_2, gae_n_enc_3 = config['gae_n_enc_1'], config['gae_n_enc_2'], config['gae_n_enc_3']
    gae_n_dec_1, gae_n_dec_2, gae_n_dec_3 = config['gae_n_dec_1'], config['gae_n_dec_2'], config['gae_n_dec_3']
    alpha_recon, alpha_cluster, alpha_hsa = config[dataset_name]['alpha_recon'], config[dataset_name]['alpha_cluster'], config[dataset_name]['alpha_hsa']

    beta = config[dataset_name]['beta']

    print("+_++++++++++")
    print(config[dataset_name]['n_superpixels'])
    dataset = HSIDataset(image_path=config[dataset_name]['img_path'],
                         gt_path=config[dataset_name]['gt_path'],
                         clip=None,
                         seg_path=None,
                         neighbors=config[dataset_name]['neighbors'],
                         patch_size=config[dataset_name]['patch_size'],
                         n_superpixels=config[dataset_name]['n_superpixels'],
                         pca=True,
                         pca_dim=config['n_pc'])
    print(dataset)
    if config['remove_bkg']:
        dataset.remove_background()
    dataset.sp_features = dataset.sp_features.to(device)
    dataset.patch_sp_features = dataset.patch_sp_features.to(device)
    dataset.sp_graph = dataset.sp_graph.to(device)
    # dataset.sadj = dataset.sadj.to(device)
    dataset.association_mat = dataset.association_mat.to(device)
    dataset.img = dataset.img.to(device)
    n_classes = dataset.n_classes
    n_samples = len(dataset)
    dataset.patch_sp_features = dataset.patch_sp_features.reshape(n_samples, -1)    # 拉成一维特征

    theoretical_max_acc(dataset.sp_labels, dataset.gt)

    # initialize model
    model = DualSPNet((dataset.sp_features.shape[1], dataset.patch_sp_features.shape[1]),
                      gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                      gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                      dataset.association_mat, device)
    model = model.to(device)

    get_parameter_number(model)

    loss_criterion = nn.MSELoss()
    loss_cluster = ClusterConsistencyLoss(n_classes, gae_n_enc_3, config[dataset_name]['temperature'], 'attention').to(device)
    loss_hsa = HardSampleAwareLoss_Pre(n_samples, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    start_time = time.time()
    best_acc = [0, 0, 0, 0, 0]
    bacc = 0
    # for epoch in range(config['n_epoches']):
    for epoch in range(100):
        print('==> Epoch: [%03d/%03d] [train]' % (epoch + 1, config['n_epoches']))
        model.train()
        optimizer.zero_grad()
        sp_feat, sp_recon, patch_sp_feat, patch_sp_recon, sp_latent_recon, pa_latent_recon = model(
            dataset.sp_features, dataset.patch_sp_features, dataset.sp_graph, dataset.sadj, dataset.img)
        recon_loss = loss_criterion(sp_feat, sp_latent_recon) + loss_criterion(patch_sp_feat, pa_latent_recon)
        clu_loss, soft_label1, soft_label2, zf, predict_labels, centers = loss_cluster(sp_feat, patch_sp_feat)
        hsa_loss, label1, label2 = loss_hsa(sp_feat, patch_sp_feat, soft_label1, soft_label2, predict_labels, centers, beta)

        loss = alpha_recon * recon_loss + alpha_cluster * clu_loss + alpha_hsa * hsa_loss


        # loss = hsa_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        print(f'loss: {loss:.4f}, recon_loss: {alpha_recon * recon_loss:.4f}, clu_loss: {alpha_cluster * clu_loss:.4f}, hsa_loss: {alpha_hsa * hsa_loss:.4f}')

        if config['remove_bkg']:
            predict_labels = dataset.recover_background(predict_labels)
        mtsf, caf, etaf = evaluate(predict_labels, dataset.association_mat, dataset.gt)
        model.eval()

        print_table([mtsf], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])

        arr = np.array([mtsf, best_acc])
        max_row_idx = np.argmax(arr[:, 0])      # 第0列是acc
        best_acc = arr[max_row_idx, :]



        if mtsf[0] > bacc:

            points = np.where(dataset.gt != 0)
            points = np.column_stack(points)
            association_mat = dataset.association_mat.numpy()
            y_pre_pixel = spixel_to_pixel_labels(predict_labels, association_mat)
            class_map = y_pre_pixel.reshape(dataset.gt.shape)
            indx = np.where(dataset.gt != 0)
            y_target = dataset.gt[indx]
            y_predict = class_map[indx]

        print('==> Epoch: [%03d/%03d] best acc:' % (epoch + 1, config['n_epoches']))
        print_table([best_acc], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])

    print(f'running time {time.time() - start_time}')

    return best_acc, time.time() - start_time


def rand_seed(seed):
    if seed == -1:
        return
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = get_config('./config.json')

    datasets = ['xuzhou']
    # datasets = ['Trento']

    # datasets = ['xuzhou']
    for dataset in datasets:
        config['dataset_name'] = dataset
        dataset_name = config['dataset_name']
        simga1s = [0.1, 1, 10, 100, 1000, 10000]
        simga2s = [0.1, 1, 10, 100, 1000, 10000]
        neighbors = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        betas = [3]
        neighbors = [50]

        for neighbor in neighbors:
            for beta in betas:
                result = []

                config[dataset_name]['neighbors'] = neighbor
                config[dataset_name]['beta'] = beta

                output_file = open("./result/{}_neighbor_result.csv".format(dataset_name), "a+", encoding="UTF-8")
                output_file.write('\n')
                output_file.write('\n')
                # output_file.write('-' * 20)
                output_file.write(
                    'num_superpixels: {}, patch_size: {}, neighbors: {}, temperature: {}, alpha_clu_loss: {}, alpha_hsa_loss: {}'.format(
                        config[dataset_name]['n_superpixels'],
                        config[dataset_name]['patch_size'],
                        config[dataset_name]['neighbors'],
                        config[dataset_name]['temperature'],
                        config[dataset_name]['alpha_cluster'],
                        config[dataset_name]['alpha_hsa']
                    ))
                output_file.write('\n')
                for seed in range(10):
                    rand_seed(seed)
                    values, runningtime = main(config)
                    result.append([values[0], values[1], values[2], values[3], values[4], runningtime])
                    output_file.write(
                        'Seed: {}, ACC: {:.2f}, NMI: {:.2f}, ARI: {:.2f}, F1: {:.2f}, Pruty: {:.2f}, RunTime:{:.4f}'.format(seed, values[0] * 100, values[1] * 100, values[2] * 100,
                                                                                                values[3] * 100, values[4] * 100, runningtime))
                    output_file.write('\n')


                result_array = np.array(result) * 100
                result_mean = np.mean(result_array, axis=0)
                result_mean = np.around(result_mean, decimals=2)
                result_std = np.std(result_array, axis=0)
                result_std = np.around(result_std, decimals=2)

                output_file.write(
                    'Average: ACC: {} ± {}, NMI: {} ± {}, ARI: {} ± {}, F1: {} ± {}, Purty: {} ± {}'.format(round(result_mean[0], 2),
                                                                                            round(result_std[0], 2),
                                                                                            round(result_mean[1], 2),
                                                                                            round(result_std[1], 2),
                                                                                            round(result_mean[2], 2),
                                                                                            round(result_std[2], 2),
                                                                                            round(result_mean[3], 2),
                                                                                            round(result_std[3], 2),
                                                                                            round(result_mean[4], 2),
                                                                                            round(result_std[4], 2),
                                                                                            ))

                output_file.write('\n')


