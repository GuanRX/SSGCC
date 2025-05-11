import torch
import torch.nn as nn
import torch.nn.functional as F
from .gae import IGAE_encoder, IGAE_decoder
from .unetpp import UNetPlusPlus


def pad_image(image):
    _, h, w = image.size()
    new_h = 2 ** (h - 1).bit_length()
    new_w = 2 ** (w - 1).bit_length()
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    image = F.pad(image, padding)
    return image


def crop_image(padded_image, original_shape):
    _, h, w = original_shape
    new_h = 2 ** (h - 1).bit_length()
    new_w = 2 ** (w - 1).bit_length()
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    crop_h = new_h - pad_top - (pad_h - pad_top)
    crop_w = new_w - pad_left - (pad_w - pad_left)
    cropped_image = padded_image[:, pad_top:pad_top+crop_h, pad_left:pad_left+crop_w]
    return cropped_image


class PatchEncoder(nn.Module):
    def __init__(self, in_channel):
        super(PatchEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (1, 1), stride=1, padding=0),  # # N*(W-2)*(W-2)*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
class PatchDecoder(nn.Module):
    def __init__(self, in_channel):
        super(PatchDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channel, (1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.decoder(x)

class SelfExpressiveLayer(nn.Module):
    def __init__(self, num_samples, init_affinity=None, device='cpu'):
        super(SelfExpressiveLayer, self).__init__()
        self.init_affinity = init_affinity
        self.device = device
        self.affinity_mat = nn.Parameter(torch.ones(num_samples, num_samples), requires_grad=True).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.affinity_mat, a=math.sqrt(5))
        if self.init_affinity is not None:
            if not isinstance(self.init_affinity, torch.Tensor):
                self.init_affinity = torch.from_numpy(self.init_affinity).float()
            self.affinity_mat.data = self.init_affinity
        else:
            nn.init.constant_(self.affinity_mat, 1.)

    def forward(self, x):
        latent_recon = torch.matmul(self.affinity_mat, x)
        return latent_recon


class DualSPNet(nn.Module):

    def __init__(self, in_channels, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, association_mat, device):
        super(DualSPNet, self).__init__()
        sp_in_c, patch_sp_in_c = in_channels
        self.association_mat = association_mat
        self.norm_association_mat = self.association_mat / self.association_mat.sum(dim=0)
        # self.sp_encoder = nn.Linear(sp_in_c, gae_n_enc_3)
        # self.sp_decoder = nn.Linear(gae_n_enc_3, sp_in_c)
        # self.patch_sp_encoder = nn.Linear(patch_sp_in_c, gae_n_dec_1)
        # self.patch_sp_decoder = nn.Linear(gae_n_dec_1, patch_sp_in_c)

        self.encoder = UNetPlusPlus(10)      # 对整张图像进行卷积
        self.sp_encoder       = IGAE_encoder(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, sp_in_c)    # 被简化了，只有一层
        self.sp_decoder       = IGAE_decoder(gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, sp_in_c)    # 被简化了，只有一层
        self.patch_sp_encoder = IGAE_encoder(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, patch_sp_in_c)
        self.patch_sp_decoder = IGAE_decoder(gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, patch_sp_in_c)
        # self.patch_sp_encoder = PatchEncoder(patch_sp_in_c)
        self.patch_sp_decoder_tmp = PatchDecoder(patch_sp_in_c)

        self.sp_linear       = nn.Sequential(nn.Linear(gae_n_enc_3, gae_n_enc_3),
                                            nn.BatchNorm1d(gae_n_enc_3),
                                            nn.Sigmoid(),
                                            nn.Dropout(0.5),
                                            )
        self.patch_sp_linear = nn.Sequential(nn.Linear(gae_n_enc_3, gae_n_enc_3),
                                            nn.BatchNorm1d(gae_n_enc_3),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            )
        
        self.self_expr_layer_sp = SelfExpressiveLayer(association_mat.shape[1], init_affinity=None, device=device)  # self.init_affinity, patch_size[0]*patch_size[1]*32,
        self.self_expr_layer_pa = SelfExpressiveLayer(association_mat.shape[1], init_affinity=None, device=device)

        self.sp_linear1 = nn.Sequential(nn.Linear(64, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True),
                                     # nn.Dropout(0.5)
                                     )
        # self.sp_lin2 = nn.Sequential(nn.Linear(512, 64),
        #                              nn.BatchNorm1d(64),
        #                              nn.ReLU(inplace=True),
        #                              nn.Dropout(0.5),
        #                              )

    def forward_img(self, x):
        c, h, w = x.size()
        x = pad_image(x)
        x = x.unsqueeze(0)
        px_feat = self.encoder(x)
        px_feat = crop_image(px_feat[0], (c, h, w))
        px_feat = px_feat.reshape(64, -1).transpose(0, 1)
        px_to_sp = torch.matmul(self.norm_association_mat.t(), px_feat)     # n_spixel * n_dim    由像素特征转来的超像素特征
        px_to_sp = self.sp_linear1(px_to_sp)                                # n_spixel * n_dim
        px_to_sp_recon = self.self_expr_layer_pa(px_to_sp)

        sp_to_pixel = px_to_sp_recon.view((-1, 512, 1, 1))
        sp_to_pixel = F.interpolate(sp_to_pixel, size=(9, 9), mode='bilinear', align_corners=True)
        patch_sp_recon = self.patch_sp_decoder_tmp(sp_to_pixel)

        return patch_sp_recon, px_to_sp, px_to_sp_recon

    def forward(self, sp_features, patch_sp_features, sp_graph, patch_sp_graph, img):
        sp_feat, z_igae_adj1 = self.sp_encoder(sp_features, sp_graph)
        sp_latent_recon = self.self_expr_layer_sp(sp_feat)
        # sp_latent_drop = self.sp_linear(sp_feat)
        sp_recon, z_hat_adj1  = self.sp_decoder(sp_latent_recon, sp_graph)

        # patch_sp_recon, patch_sp_feat, pa_latent_recon = self.forward_img(img)

        patch_sp_feat, z_igae_adj2 = self.patch_sp_encoder(patch_sp_features, sp_graph) # 换成sp_graph精度会跌
        pa_latent_recon = self.self_expr_layer_pa(patch_sp_feat)
        # pa_latent_drop = self.patch_sp_linear(patch_sp_feat)
        patch_sp_recon, z_hat_adj2 = self.patch_sp_decoder(pa_latent_recon, sp_graph)



        return sp_feat, sp_recon, patch_sp_feat, patch_sp_recon, sp_latent_recon, pa_latent_recon



