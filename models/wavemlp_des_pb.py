from functools import reduce

import torch
import torch.nn as nn

from timm.models.layers import DropPath

import torch.nn.functional as F
from utils.pos_embed import exchange_token, exchange_patch, get_mask_box, jigsaw_token, cutout_patch, erase_patch, \
    mixup_patch, jigsaw_patch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        #         x_1=self.fc_h(x)
        #         x_2=self.fc_w(x)
        #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)  # token mixer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.channel_att = SELayer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.channel_att(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SELayer(nn.Module):
    def __init__(self, channel, down=False, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.down = down
        if down:
            self.conv = nn.Conv2d(channel, channel // 2, 1, 1, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # 对应Excitation操作
        if self.down:
            return self.conv(x * y.expand_as(x))
        else:
            return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, in_chans, embed_dim, kernel_size, stride, padding):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.norm(x)
        return x


def basic_blocks(dim, num_layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, mode='fc', **kwargs):
    blocks = []
    for block_idx in range(num_layers):
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, drop_path=drop_path_rate, norm_layer=norm_layer, mode=mode))
    blocks = nn.Sequential(*blocks)
    return blocks


class Vec2Patch(nn.Module):
    def __init__(self, in_chans, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * in_chans
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        # b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)
        # feat = b c h w
        return feat


class Wave_block(nn.Module):
    def __init__(self, cfg, in_chans, embed_dim, kernel_size, padding, stride, use_feature_mask):
        super(Wave_block, self).__init__()
        output_size = (14, 14)
        num_layers = cfg.MODEL.TRANSFORMER.ENCODER_LAYERS
        dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE
        self.add_shortcut = cfg.MODEL.TRANSFORMER.USE_LOCAL_SHORTCUT
        self.concat_short = cfg.MODEL.TRANSFORMER.USE_CONCAT_SHORTCUT
        self.use_feature_mask = use_feature_mask
        self.patch2vec = PatchEmbed(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.vec2patch = Vec2Patch(in_chans, embed_dim, output_size, kernel_size, stride, padding)
        self.wave_mlp = basic_blocks(dim=embed_dim * 2, num_layers=num_layers, drop_path_rate=dropout)
        self.use_feature_mask = cfg.MODEL.TRANSFORMER.USE_FEATURE_MASK
        if self.concat_short:
            self.SEnet = SELayer(channel=embed_dim * 2, down=True)

    def forward(self, x):
        feat, mask = x
        trans_feat = self.patch2vec(feat)
        if self.training and self.use_feature_mask:
            if self.feature_aug_type == 'exchange_token':
                feature_mask = exchange_token()
                trans_feat = feature_mask(trans_feat, mask)
            elif self.feature_aug_type == 'cutout_patch':
                feature_mask = cutout_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'erase_patch':
                feature_mask = erase_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'mixup_patch':
                feature_mask = mixup_patch()
                trans_feat = feature_mask(trans_feat)

        if self.use_feature_mask:
            if self.feature_aug_type == 'jigsaw_patch':
                feature_mask = jigsaw_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'jigsaw_token':
                feature_mask = jigsaw_token()
                trans_feat = feature_mask(trans_feat)
        trans_feat = self.vec2patch(trans_feat)
        trans_feat = self.wave_mlp(trans_feat)

        if self.add_shortcut:
            trans_feat = feat + trans_feat
        elif self.concat_short:
            trans_feat = torch.concat([feat, trans_feat], dim=1)
            trans_feat = self.SEnet(trans_feat)
        return trans_feat


class Wave_blocks(nn.Module):
    def __init__(self, cfg, blocks_name, kernel_list, use_feature_mask):
        super(Wave_blocks, self).__init__()
        dim = cfg.MODEL.TRANSFORMER.DIM_MODEL
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE
        self.use_feature_mask = use_feature_mask
        self.use_output_layer = cfg.MODEL.TRANSFORMER.USE_OUTPUT_LAYER
        self.use_global_shortcut = cfg.MODEL.TRANSFORMER.USE_GLOBAL_SHORTCUT
        self.blocks_name = blocks_name
        self.scale_size = len(self.blocks_name)
        kernels = {
            (1, 1): [(0, 0), (1, 1)],
            (3, 3): [(1, 1), (1, 1)]
        }
        strides = []
        paddings = []
        for size in kernel_list:
            if size not in [(1, 1), (3, 3)]:
                raise ValueError('Undefined kernel size.')
            paddings.append(kernels[size][0])
            strides.append(kernels[size][1])
        self.blocks = nn.ModuleDict()
        # block = Wave_block(cfg, in_chans=dim,
        #                   embed_dim=dim//2,
        #                  kernel_size=kernel
        #                  , stride=stride, padding=padding)
        for name, kernel, stride, padding in zip(self.blocks_name, kernel_list,
                                                 strides, paddings):
            block = Wave_block(cfg, in_chans=dim // self.scale_size,  # 512//2
                               embed_dim=dim // (self.scale_size * 2),  # 512//4
                               kernel_size=kernel
                               , stride=stride,
                               padding=padding,
                               use_feature_mask=use_feature_mask)
            self.blocks[name] = nn.Sequential(block)
        if self.use_output_layer:
            self.output_linear = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.mask_para = [cfg.MODEL.MASK_SHAPE, cfg.MODEL.MASK_SIZE, cfg.MODEL.MASK_MODE]

    def forward(self, x):
        feat_, mask = x
        feat_list = []
        if self.training and self.use_feature_mask and self.feature_aug_type == 'exchange_patch':
            feature_mask = exchange_patch(self.mask_para[0], self.mask_para[1], self.mask_para[2])
            feat_ = feature_mask(feat_)
        # 14*14*512
        # feat = self.block((feat_,mask)

        for name, feat in zip(self.blocks_name, torch.chunk(feat_, len(self.blocks_name), dim=1)):
            feat = self.blocks[name]((feat, mask))
            feat_list.append(feat)
        feat = torch.cat(feat_list, 1)
        if self.use_output_layer:
            feat = self.output_linear(feat)
        if self.use_global_shortcut:
            feat = feat + feat_
        return feat


class WaveHead(nn.Module):
    def __init__(self, cfg, block_name, kernel_size, use_feature_mask):
        super(WaveHead, self).__init__()
        dim = cfg.MODEL.TRANSFORMER.DIM_MODEL
        self.use_feature_mask = use_feature_mask
        mask_shape = cfg.MODEL.MASK_SHAPE
        mask_size = cfg.MODEL.MASK_SIZE
        mask_mode = cfg.MODEL.MASK_MODE
        self.bypass_mask = exchange_patch(mask_shape, mask_size, mask_mode)
        self.get_mask_box = get_mask_box(mask_shape, mask_size, mask_mode)
        self.encoder = Wave_blocks(
            cfg=cfg,
            blocks_name=block_name,
            kernel_list=kernel_size,
            use_feature_mask=use_feature_mask
        )
        self.skip = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.conv1 = nn.Conv2d(1024, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim, 2048, 1, 1, 0)

    def forward(self, x):
        mask = self.get_mask_box(x)
        if self.use_feature_mask:
            skip = self.skip(x)
            if self.training:
                skip = self.bypass_mask(skip)
        else:
            skip = x
        feats = {}
        feats['before_trans'] = F.adaptive_max_pool2d(skip, 1)
        x = self.conv1(x)
        x = self.encoder((x, mask))
        x = self.conv2(x)
        feats['after_trans'] = F.adaptive_max_pool2d(x, 1)
        return feats


class WaveHead_two(nn.Module):
    def __init__(self, cfg, block_name, kernel_size, use_feature_mask):
        super(WaveHead_two, self).__init__()
        dim = cfg.MODEL.TRANSFORMER.DIM_MODEL
        self.use_feature_mask = use_feature_mask
        mask_shape = cfg.MODEL.MASK_SHAPE
        mask_size = cfg.MODEL.MASK_SIZE
        mask_mode = cfg.MODEL.MASK_MODE
        self.bypass_mask = exchange_patch(mask_shape, mask_size, mask_mode)
        self.get_mask_box = get_mask_box(mask_shape, mask_size, mask_mode)
        self.reg_head = NonLocalBlock(1024)
        self.encoder = Wave_blocks(
            cfg=cfg,
            blocks_name=block_name,
            kernel_list=kernel_size,
            use_feature_mask=use_feature_mask
        )
        self.skip = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.conv1 = nn.Conv2d(1024, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim, 2048, 1, 1, 0)

    def forward(self, x):
        mask = self.get_mask_box(x)
        if self.use_feature_mask:
            skip = self.skip(x)
            if self.training:
                skip = self.bypass_mask(skip)
        else:
            skip = x
        feats = {}
        y = self.reg_head(x)
        y = F.adaptive_max_pool2d(y, 1)
        feats['before_trans'] = F.adaptive_max_pool2d(skip, 1)
        x = self.conv1(x)
        x = self.encoder((x, mask))
        x = self.conv2(x)
        feats['after_trans'] = F.adaptive_max_pool2d(x, 1)
        return feats, y


class WaveHead_des(nn.Module):
    def __init__(self, cfg, block_name, kernel_size, use_feature_mask):
        super(WaveHead_des, self).__init__()
        dim = cfg.MODEL.TRANSFORMER.DIM_MODEL
        self.use_feature_mask = use_feature_mask
        mask_shape = cfg.MODEL.MASK_SHAPE
        mask_size = cfg.MODEL.MASK_SIZE
        mask_mode = cfg.MODEL.MASK_MODE
        self.bypass_mask = exchange_patch(mask_shape, mask_size, mask_mode)
        self.get_mask_box = get_mask_box(mask_shape, mask_size, mask_mode)
        self.reg_head = NonLocalBlock(1024)
        self.encoder = Wave_blocks(
            cfg=cfg,
            blocks_name=block_name,
            kernel_list=kernel_size,
            use_feature_mask=use_feature_mask
        )
        self.skip = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.conv1 = nn.Conv2d(1024, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim, 2048, 1, 1, 0)

        # self.share_conv = nn.Sequential(nn.Conv2d(1024, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(),
        #                                SELayer(dim), nn.Conv2d(dim, 1024, 1, 1, 0), nn.BatchNorm2d(1024), nn.ReLU()
        #                               )
        self.pobe = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 256), nn.ReLU(inplace=True)
        )
        self.reid_pobe = nn.Sequential(nn.Linear(256, 2),
                                       nn.Softmax(dim=1),
                                       )
        self.cls_pobe = nn.Sequential(nn.Linear(256, 2),
                                      nn.Softmax(dim=1),
                                      )
        self.mm_convs = nn.Sequential(nn.Conv2d(1024, dim, 3, 1, 1),
                                      nn.BatchNorm2d(dim), nn.ReLU(),
                                      )
        self.mm_fcs = nn.Sequential(
                                    nn.Linear(dim*14*14, 2*256),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        # x = self.share_conv(x)
        # x_mask = torch.mean(x,dim=0,keepdim=True)
        # x_mask = self.mm_convs(x_mask).flatten(1)
        # x_mask = self.mm_fcs(x_mask)*2.
        # mask_c = torch.chunk(x_mask,2,1)

        aa = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        reid_p = self.reid_pobe(self.pobe(aa)).view(-1, 1, 2)
        cls_p = self.cls_pobe(self.pobe(aa)).view(-1, 1, 2)

        mask = self.get_mask_box(x)
        if self.use_feature_mask:  # 使用mask
            skip = self.skip(x)
            if self.training:
                skip = self.bypass_mask(skip)
        else:
            skip = x
        feats,feats_ = {},{}#

        y = self.reg_head(x)

        y = F.adaptive_max_pool2d(y, 1)
        feats['before_trans'] = F.adaptive_max_pool2d(skip, 1)
        feats_['before_trans'] = F.adaptive_avg_pool2d(skip,1)#
        x = self.conv1(x)
        x = self.encoder((x, mask))
        x = self.conv2(x)
        feats['after_trans'] = F.adaptive_max_pool2d(x, 1)
        feats_['after_trans'] = F.adaptive_avg_pool2d(x,1)#
        return feats, y, reid_p, cls_p, feats_


class get_mask_des(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(get_mask_des, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(out_chans * 14 * 14, out_chans)

    def forward(self, x):
        x_mask = torch.mean(x, dim=0, keepdim=True)
        x_mask = self.norm(self.conv(x_mask))
        x_mask = x_mask.flatten(1)
        mask_cls = self.act(self.fc(x_mask)) * 2.
        mask_c = torch.chunk(mask_cls, 2, dim=1)
        return mask_c
