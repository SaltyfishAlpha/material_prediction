import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum, Tensor
from torchvision import transforms
import torchvision
import cv2 as cv
import scipy.misc


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H*W*self.in_channels*self.kernel_size**2/self.stride**2
        flops += H*W*self.in_channels*self.out_channels
        return flops


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, H, W):
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 3
        return flops


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 5
        return flops

    #########################################


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.se_layer = SELayer(dim)
        self.ll = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        self.sigmoid = nn.Sigmoid()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,
                dino_mat, point_feature, color_feature, shadow_detect_feature, normal, shadow_mask,
                cluster_mask,
                attn_kv=None, mask=None):
        B_, N, C = x.shape
        # x = self.se_layer(x)
        # one = torch.ones_like(xm)
        # zero = torch.zeros_like(xm)
        # xm = torch.where(xm < 0.1, one, one*2)

        # if you don't add MAT or DINO, the dino_mat will be 1
        # like LayerNorm
        if shadow_detect_feature != None:
            if shadow_detect_feature.shape[-1] != 1:
                shadow_detect_feature = shadow_detect_feature.unsqueeze(2)
                normalizer = torch.sqrt(
                    (shadow_detect_feature @ shadow_detect_feature.transpose(-2, -1)).squeeze(-2)).detach()
                normalizer = torch.clamp(normalizer, 1.0e-20, 1.0e10)
                shadow_detect_feature = shadow_detect_feature.squeeze(2) / normalizer

                # 1-correlation
            shadow_correlation_map = shadow_detect_feature @ shadow_detect_feature.transpose(-2, -1).contiguous()
            # this not need, because shadow_detect_feature has through sigmoid layer
            shadow_correlation_map = torch.clamp(shadow_correlation_map, 0.0, 1.0)
            shadow_correlation_map = 1 - shadow_correlation_map
            shadow_correlation_map = torch.unsqueeze(shadow_correlation_map, dim=1)

            # one = torch.ones_like(shadow_detect_feature)
            # shadow_detect_feature = torch.where(shadow_detect_feature < 0.1, one, one*2)
            # shadow_correlation_map = shadow_detect_feature @ shadow_detect_feature.transpose(-2, -1)
            # one = torch.ones_like(shadow_correlation_map)
            # shadow_correlation_map = torch.where(shadow_correlation_map==2, one, one*0.2)
            # shadow_correlation_map = torch.unsqueeze(shadow_correlation_map, dim=1)

        # when MAT network is ok, this part will be replace
        if dino_mat != None:
            dino_mat = dino_mat.unsqueeze(2)
            normalizer = torch.sqrt((dino_mat @ dino_mat.transpose(-2, -1)).squeeze(-2)).detach()
            normalizer = torch.clamp(normalizer, 1.0e-20, 1.0e10)
            dino_mat = dino_mat.squeeze(2) / normalizer
            dino_mat_correlation_map = dino_mat @ dino_mat.transpose(-2, -1).contiguous()
            dino_mat_correlation_map = torch.clamp(dino_mat_correlation_map, 0.0, 1.0e10)
            dino_mat_correlation_map = torch.unsqueeze(dino_mat_correlation_map, dim=1)

        if point_feature != None:
            point_feature = point_feature.unsqueeze(2)
            Point = point_feature.repeat(1, 1, self.win_size[0] * self.win_size[1], 1)
            Point = Point - Point.transpose(-2, -3)
            normal = normal.unsqueeze(2).repeat(1, 1, self.win_size[0] * self.win_size[1], 1)
            Point = Point * normal
            Point = torch.abs(torch.sum(Point, dim=3))

            plane_correlation_map = 0.5 * (Point + Point.transpose(-1, -2))
            plane_correlation_map = plane_correlation_map.unsqueeze(1)
            plane_correlation_map = torch.exp(-plane_correlation_map)

        if shadow_mask != None:
            one = torch.ones_like(shadow_mask)
            shadow_mask = torch.where(shadow_mask < 0.1, one, one * 2)
            mm = shadow_mask @ shadow_mask.transpose(-2, -1)
            one = torch.ones_like(mm)
            mm = torch.where(mm == 2, one, one * 0.2)
            mm = torch.unsqueeze(mm, dim=1)

        if cluster_mask != None:
            cluster_correlation_map = cluster_mask @ cluster_mask.transpose(-2, -1)
            cluster_correlation_map = cluster_correlation_map.unsqueeze(1)

        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # print(x.shape, q.shape, k.shape, v.shape, attn.shape)
        # print('cluster shape:', cluster_mask.shape)

        if shadow_detect_feature != None:
            attn = shadow_correlation_map * attn
        if dino_mat != None:
            attn = dino_mat_correlation_map * attn
        if point_feature != None:
            attn = plane_correlation_map * attn
        if cluster_mask != None:
            attn = cluster_correlation_map * attn
        # if shadow_detect_feature != None:
        #     for i in range(0, shadow_correlation_map.shape[0]):
        #         win_shadow_detect_feature = shadow_correlation_map.permute(0,2,3,1).detach().cpu().numpy()[i]
        #         win_dino_mat = dino_mat_correlation_map.permute(0,2,3,1).detach().cpu().numpy()[i]
        #         win_point_feature = plane_correlation_map.permute(0,2,3,1).detach().cpu().numpy()[i]
        #         cv.imwrite(f"/home/disk1/ShadowFormer/correlation/shadow_detect_feature_{i}.png", 255 * win_shadow_detect_feature)
        #         cv.imwrite(f"/home/disk1/ShadowFormer/correlation/dino_mat_{i}.png", 255 * win_dino_mat)
        #         cv.imwrite(f"/home/disk1/ShadowFormer/correlation/point_feature_{i}.png", 255 * win_point_feature)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # weight = torch.ones((attn.shape[0],1,attn.shape[2],attn.shape[3])).cuda()
        # if shadow_detect_feature != None:
        #     weight *= shadow_correlation_map
        # if dino_mat != None:
        #     weight *= dino_mat_correlation_map
        # if point_feature != None:
        #     weight *= plane_correlation_map

        # weight = weight.reshape(-1,1,attn.shape[2]*attn.shape[3])
        # weight /= torch.clamp(torch.sum(weight, dim=-1).unsqueeze(-1).detach(), min = 1.0e-10, max = 1.0e10)
        # weight = weight.reshape(-1,1,attn.shape[2],attn.shape[3])
        # attn = weight * attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.ll(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H, W)
        # attn = (q @ k.transpose(-2, -1))
        if self.token_projection != 'linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N * 2
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * 2 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### SIM Transformer #############
class SIMTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution=(256,256), win_size=10, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(
            self, x,
            # dino_mat=None, point=None, color=None, shadow_detect_feature=None, normal=None, shadow_mask=None,
            cluster_mask=None,
            mask=None,
            # img_size=(128, 128)):
            ):
        B, L, C = x.shape
        H = self.input_resolution[0]
        W = self.input_resolution[1]
        assert L == W * H, f"Input image size ({H}*{W} doesn't match model ({L})."

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1).contiguous()
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x
        x = self.norm1(x)

        # torch.cuda.synchronize()
        # time1 = time.time()
        x = x.view(B, H, W, C)

        # if dino_mat != None:
        #     C_dino_mat = dino_mat.shape[1]
        # if shadow_detect_feature != None:
        #     C_shadow_detect_feature = shadow_detect_feature.shape[1]
        # if point != None:
        #     C_point = point.shape[1]
        # if normal != None:
        #     C_normal = normal.shape[1]
        # if color != None:
        #     C_color = color.shape[1]
        # if shadow_mask != None:
        #     C_shadow_mask = shadow_mask.shape[1]

        # if dino_mat != None:
        #     dino_mat = dino_mat.permute(0, 2, 3, 1).contiguous()
        #
        # if shadow_detect_feature != None:
        #     shadow_detect_feature = shadow_detect_feature.permute(0, 2, 3, 1).contiguous()
        #
        # if point != None:
        #     point = point.permute(0, 2, 3, 1).contiguous()
        #
        # if normal != None:
        #     normal = normal.permute(0, 2, 3, 1).contiguous()
        #
        # if color != None:
        #     color = color.permute(0, 2, 3, 1).contiguous()
        #
        # if shadow_mask != None:
        #     shadow_mask = shadow_mask.permute(0, 2, 3, 1).contiguous()

        if cluster_mask is not None:
            cluster_mask = cluster_mask.permute(0, 2, 3, 1).contiguous()
            C_cluster_mask = 1

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # if dino_mat != None:
            #     shifted_dino_mat = torch.roll(dino_mat, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # if shadow_detect_feature != None:
            #     shifted_shadow = torch.roll(shadow_detect_feature, shifts=(-self.shift_size, -self.shift_size),
            #                                 dims=(1, 2))
            # if point != None:
            #     shifted_point = torch.roll(point, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # if normal != None:
            #     shifted_normal = torch.roll(normal, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # if color != None:
            #     shifted_color = torch.roll(color, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # if shadow_mask != None:
            #     shifted_shadow_mask = torch.roll(shadow_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if cluster_mask != None:
                shifted_cluster_mask = torch.roll(cluster_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

            # if dino_mat != None:
            #     shifted_dino_mat = dino_mat
            # if shadow_detect_feature != None:
            #     shifted_shadow = shadow_detect_feature
            # if point != None:
            #     shifted_point = point
            # if normal != None:
            #     shifted_normal = normal
            # if color != None:
            #     shifted_color = color
            # if shadow_mask != None:
            #     shifted_shadow_mask = shadow_mask
            if cluster_mask != None:
                shifted_cluster_mask = cluster_mask

        # torch.cuda.synchronize()
        # time2 = time.time()
        # print(f"shift_time:{time2-time1}")

        # partition windows
        # torch.cuda.synchronize()
        # time1 = time.time()

        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        shadow_windows = None
        # if shadow_detect_feature != None:
        #     shadow_windows = window_partition(shifted_shadow, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     shadow_windows = shadow_windows.view(-1, self.win_size * self.win_size,
        #                                          C_shadow_detect_feature)  # nW*B, win_size*win_size, C

        dino_mat_windows = None
        # if dino_mat != None:
        #     dino_mat_windows = window_partition(shifted_dino_mat, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     dino_mat_windows = dino_mat_windows.view(-1, self.win_size * self.win_size,
        #                                              C_dino_mat)  # nW*B, win_size*win_size, C

        point_windows = None
        # if point != None:
        #     point_windows = window_partition(shifted_point, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     point_windows = point_windows.view(-1, self.win_size * self.win_size, C_point)  # nW*B, win_size*win_size, C

        normal_windows = None
        # if normal != None:
        #     normal_windows = window_partition(shifted_normal, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     normal_windows = normal_windows.view(-1, self.win_size * self.win_size,
        #                                          C_normal)  # nW*B, win_size*win_size, C

        color_windows = None
        # if color != None:
        #     color_windows = window_partition(shifted_color, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     color_windows = color_windows.view(-1, self.win_size * self.win_size, C_color)  # nW*B, win_size*win_size, C

        shadow_mask_windows = None
        # if shadow_mask != None:
        #     shadow_mask_windows = window_partition(shifted_shadow_mask,
        #                                            self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        #     shadow_mask_windows = shadow_mask_windows.view(-1, self.win_size * self.win_size,
        #                                                    C_shadow_mask)  # nW*B, win_size*win_size, C

        cluster_mask_windows = None
        if cluster_mask != None:
            # nW*B, win_size, win_size, C  N*C->C
            cluster_mask_windows = window_partition(shifted_cluster_mask, self.win_size)
            # nW*B, win_size*win_size, C
            cluster_mask_windows = cluster_mask_windows.view(-1, self.win_size*self.win_size, C_cluster_mask)

        # torch.cuda.synchronize()
        # time2 = time.time()
        # print(f"partition_time:{time2-time1}")

        # W-MSA/SW-MSA

        # torch.cuda.synchronize()
        # time1 = time.time()
        # torch.cuda.synchronize()
        attn_windows = self.attn(x_windows,
                                 dino_mat_windows, point_windows, color_windows, shadow_windows,
                                 normal_windows, shadow_mask_windows, cluster_mask_windows,
                                 mask=attn_mask)  # nW*B, win_size*win_size, C
        # time2 = time.time()
        # print(f"atte_time:{time2 - time1}")

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)

        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32

        # torch.cuda.synchronize()
        # time1 = time.time()
        x = self.CAB(x)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print(f"CAB_time:{time2 - time1}")

        # torch.cuda.synchronize()
        # time1 = time.time()
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), img_size=self.input_resolution))
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print(f"FFN_time:{time2 - time1}")

        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        print("LeWin:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.in_features*self.hidden_features
        # fc2
        flops += H*W*self.hidden_features*self.out_features
        print("MLP:{%.2f}"%(flops/1e9))
        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128,128)):
        # bs x hw x c
        bs, hw, c = x.size()
        # hh = int(math.sqrt(hw))
        hh = img_size[0]
        ww = img_size[1]

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2).contiguous() # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

