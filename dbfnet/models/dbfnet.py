import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet
import einops
import math
from timm.models.layers import DropPath, trunc_normal_


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class Position_Encoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
		@param x: [B,N,d]
		@param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
		@param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
		@return:
		'''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(x, pe):
        """ combine feature and position code
		"""
        return Position_Encoding.embed_rotary(x, pe[..., 0], pe[..., 1])

    def forward(self, feature):
        bsize, npoint, _ = feature.shape
        position = torch.arange(npoint, device=feature.device).unsqueeze(dim=0).repeat(bsize, 1).unsqueeze(dim=-1)
        # [1, 1, d/2]
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=feature.device) * (
                -math.log(10000.0) / self.feature_dim)).view(1, 1, -1)
        sinx = torch.sin(position * div_term)  # [B, N, d//2]
        cosx = torch.cos(position * div_term)
        sinx, cosx = map(lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1), [sinx, cosx])
        position_code = torch.stack([cosx, sinx], dim=-1)
        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class DynamicSparseLinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        # 使用nn.Parameter定义可学习的阈值
        self.threshold = nn.Parameter(torch.tensor([0.1]))

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        v_length = values.size(1)
        values = values / v_length

        KV = torch.einsum("nshd,nshv->nhdv", K, values)
        attention_scores = torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))
        
        # 使用torch.where代替直接的大于操作，以保持梯度流
        # 注意：这里使用0作为不满足条件时的值
        sparse_attention_scores = torch.where(attention_scores > self.threshold, attention_scores, torch.zeros_like(attention_scores))
        Z = 1 / (sparse_attention_scores + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class TransEncoderLayer(nn.Module):
    def __init__(self,
                 query_in_dim,
                 source_in_dim,
                 token_dim,
                 num_heads):
        super(TransEncoderLayer, self).__init__()

        self.dim = query_in_dim // num_heads
        self.nhead = num_heads
        # multi-head attention
        self.q_proj = nn.Linear(query_in_dim, token_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(source_in_dim, token_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(source_in_dim, token_dim * num_heads, bias=False)
        self.attention = DynamicSparseLinearAttention()
        self.merge = nn.Linear(token_dim * num_heads, token_dim * num_heads, bias=False)
        self.rotary_emb = Position_Encoding(dim=token_dim * num_heads)
        
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(token_dim * num_heads * 2, token_dim * num_heads * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(token_dim * num_heads * 2, token_dim * num_heads, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(token_dim * num_heads)
        self.norm2 = nn.LayerNorm(token_dim * num_heads)
        self.aug_shortcut = nn.Sequential(
            nn.Linear(query_in_dim, token_dim * num_heads),
            nn.LayerNorm(token_dim * num_heads)
        )

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        mixed_query_layer = self.q_proj(query)
        mixed_key_layer = self.k_proj(key)
        que_pe = self.rotary_emb(mixed_query_layer)
        key_pe = self.rotary_emb(mixed_key_layer)
        query = Position_Encoding.embed_pos(mixed_query_layer, que_pe).view(bs, -1, self.nhead, self.dim)
        key = Position_Encoding.embed_pos(mixed_key_layer, key_pe).view(bs, -1, self.nhead, self.dim)
        
        # query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        # key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message + self.aug_shortcut(x)


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, query_in_dim=512, source_in_dim=256, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(query_in_dim, token_dim * num_heads)
        self.to_key = nn.Linear(source_in_dim, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x, source):
        query = self.to_query(x)
        key = self.to_key(source)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        message = self.Proj(G * key)
        message = torch.nn.functional.interpolate(key.transpose(1, 2), size=query.shape[1]).transpose(1, 2)
        out = message + query #BxNxD
        out = self.final(out) # BxNxD
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, query_in_dim=512, source_in_dim=256, token_dim=256, num_heads=1):
        super().__init__()
        self.attn = EfficientAdditiveAttnetion(query_in_dim, source_in_dim, token_dim, 1)
        self.linear = Mlp(in_features=query_in_dim, hidden_features=int(query_in_dim * 2))
    
    def forward(self, x, source):
        x = self.attn(x, source) + x
        x = torch.nn.functional.normalize(x, dim=-1) #BxNxD
        x = self.linear(x) + x
        return x


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = torch.cat([x, residual], dim=1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = torch.cat([2 * x * wei, 2 * residual * (1 - wei)], dim=1)
        return xo


class DBFNet(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, n_kps=8
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        cnn = psp_models['resnet34'.lower()]()

        rndla = RandLANet(rndla_cfg)

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        self.rndla_pre_stages = rndla.fc0
        
        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        
        self.ds_sr = [4, 8, 8, 8]
        self.rndla_ds_stages = rndla.dilated_res_blocks
        self.ds_rgb_oc = [64, 128, 512, 1024]
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        
        self.ds_p2r_fuse_layer = nn.ModuleList()
        self.ds_r2p_fuse_layer = nn.ModuleList()
        for i in range(4):
            self.ds_p2r_fuse_layer.append(
                TransEncoderLayer(self.ds_rgb_oc[i], self.ds_rndla_oc[i], token_dim=self.ds_rgb_oc[i] // 4, num_heads=4)
            )
            self.ds_r2p_fuse_layer.append(
                TransEncoderLayer(self.ds_rndla_oc[i], self.ds_rgb_oc[i], token_dim=self.ds_rndla_oc[i] // 4, num_heads=4)
            )

        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(n_fuse_layer):
            self.up_fuse_p2r_fuse_layers.append(
                EfficientAdditiveAttnetion(self.up_rgb_oc[i], self.up_rndla_oc[i], token_dim=self.up_rgb_oc[i], num_heads=1)
            )
            self.up_fuse_r2p_fuse_layers.append(
                EfficientAdditiveAttnetion(self.up_rndla_oc[i], self.up_rgb_oc[i], token_dim=self.up_rndla_oc[i], num_heads=1)
            )
        
        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.
        # self.up_rndla_oc[-1] + self.up_rgb_oc[-1]

        self.fuse_layer = EnhancedDenseFusion()

        self.rgbd_seg_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(128)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*3, activation=None)
        )
    
    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features
      
    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features
    
    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features
    
    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # stride = 2, [bs, c, 240, 320]

        # rndla pre
        xyz, p_emb = self._break_up_pc(inputs['cld_rgb_nrm'])
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb)
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1
        
        # ###################### encoding stages #############################
        ds_emb = []
        
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()
            
            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )
            f_sampled_i = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds])
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)
            
            # cross attenrion for rgb and point embedding
            rgb_emb0 = rgb_emb0.view(bs, c, -1).transpose(1, 2)
            p_emb0 = p_emb0.squeeze(dim=-1).transpose(1, 2)
            
            rgb_emb = self.ds_p2r_fuse_layer[i_ds](rgb_emb0, p_emb0).transpose(1, 2).view(bs, c, hr, wr)
            p_emb = self.ds_r2p_fuse_layer[i_ds](p_emb0, rgb_emb0).transpose(1, 2).unsqueeze(dim=3)
            
            ds_emb.append(p_emb)

        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()
            
            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i
            
            # cross attenrion for rgb and point embedding
            rgb_emb0 = rgb_emb0.view(bs, c, -1).transpose(1, 2)
            p_emb0 = p_emb0.squeeze(dim=-1).transpose(1, 2)
            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](rgb_emb0, p_emb0).transpose(1, 2).view(bs, c, hr, wr)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](p_emb0, rgb_emb0).transpose(1, 2).unsqueeze(dim=3)

        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)

        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        # rgbd_emb = self.fusion_layer(rgb_emb, pcld_emb)

        rgbd_emb = self.fuse_layer(rgb_emb_c, p_emb)
        # Use simple concatenation. Good enough for fully fused RGBD feature.
        # rgbd_emb = torch.cat([rgb_emb_c, p_emb, fuse_emb], dim=1)

        # ###################### prediction stages #############################
        rgbd_segs = self.rgbd_seg_layer(rgbd_emb)
        pred_kp_ofs = self.kp_ofst_layer(rgbd_emb)
        pred_ctr_ofs = self.ctr_ofst_layer(rgbd_emb)

        pred_kp_ofs = pred_kp_ofs.view(
            bs, self.n_kps, 3, -1
        ).permute(0, 1, 3, 2).contiguous()
        pred_ctr_ofs = pred_ctr_ofs.view(
            bs, 1, 3, -1
        ).permute(0, 1, 3, 2).contiguous()

        # return rgbd_seg, pred_kp_of, pred_ctr_of
        end_points['pred_rgbd_segs'] = rgbd_segs
        end_points['pred_kp_ofs'] = pred_kp_ofs
        end_points['pred_ctr_ofs'] = pred_ctr_ofs

        return end_points
    

# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(32, 256, 1)

        self.conv3 = torch.nn.Conv1d(96, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1)  # 96+ 512 + 1024 = 1632


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels, 1)
        self.key_conv = nn.Conv1d(channels, channels, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.attn = DynamicSparseLinearAttention()
        self.dim = channels // 4
        self.nhead = 4

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(channels * 2, channels, bias=False),
        )
        # norm and dropout
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.aug_shortcut = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        bs = x.size(0)
        proj_query = self.query_conv(x).transpose(1, 2).view(bs, -1, self.nhead, self.dim)
        proj_key = self.key_conv(x).transpose(1, 2).view(bs, -1, self.nhead, self.dim)
        proj_value = self.value_conv(x).transpose(1, 2).view(bs, -1, self.nhead, self.dim)
        message = self.attn(proj_query, proj_key, proj_value).view(bs, -1, self.nhead*self.dim).transpose(1, 2)
        message = self.norm1(message.transpose(1, 2))
        
        message = self.mlp(torch.cat([x.transpose(1, 2), message], dim=2))
        message = self.norm2(message).transpose(1, 2)

        return message + x + self.aug_shortcut(x)


class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels, 1)
        self.key_conv = nn.Conv1d(channels, channels, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.attn = DynamicSparseLinearAttention()
        self.dim = channels // 4
        self.nhead = 4
        
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(channels * 2, channels, bias=False),
        )
        # norm and dropout
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.aug_shortcut = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x1, x2):
        bs = x1.size(0)
        proj_query = self.query_conv(x1).transpose(1, 2).view(bs, -1, self.nhead, self.dim)
        proj_key = self.key_conv(x2).transpose(1, 2).view(bs, -1, self.nhead, self.dim)
        proj_value = self.value_conv(x2).transpose(1, 2).view(bs, -1, self.nhead, self.dim)

        message = self.attn(proj_query, proj_key, proj_value).view(bs, -1, self.nhead*self.dim).transpose(1, 2)
        message = self.norm1(message.transpose(1, 2))
        
        message = self.mlp(torch.cat([x1.transpose(1, 2), message], dim=2))
        message = self.norm2(message).transpose(1, 2)

        return message + x1 + self.aug_shortcut(x1)


class EnhancedDenseFusion(nn.Module):
    def __init__(self):
        super(EnhancedDenseFusion, self).__init__()

        self.self_att_rgb = SelfAttention(64)
        self.self_att_cld = SelfAttention(64)
        self.cross_att_rgb = CrossAttention(64)
        self.cross_att_pcl = CrossAttention(64)

    def forward(self, rgb_emb, cld_emb):

        rgb = self.self_att_rgb(rgb_emb)
        cld = self.self_att_cld(cld_emb)
        cross_rgb = self.cross_att_rgb(rgb, cld)
        cross_cld = self.cross_att_pcl(cld, rgb)
        feat = torch.cat((cross_rgb, cross_cld), dim=1)

        return feat


def main():
    from common import ConfigRandLA
    rndla_cfg = ConfigRandLA

    n_cls = 22
    model = DBFNet(n_cls, rndla_cfg.num_points, rndla_cfg)
    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )


if __name__ == "__main__":
    main()
