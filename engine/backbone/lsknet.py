import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from ..core import register

__all__ = ['LSKNet']

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def trunc_normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class LayerNorm2d(nn.Module):
    """LayerNorm that operates directly on NCHW tensors."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim, lsk_use_3x3=False):
        super().__init__()
        if lsk_use_3x3:
            self.conv0 = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            )
        else:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn = torch.max(attn, dim=1, keepdim=True)[0]
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()
        sig1, sig2 = sig.chunk(2, dim=1)
        attn = attn1 * sig1 + attn2 * sig2

        attn = self.conv(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model, lsk_use_3x3=False):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model, lsk_use_3x3=lsk_use_3x3)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None, lsk_use_3x3=False):
        super().__init__()
        if norm_cfg and norm_cfg.get('type') == 'SyncBN':
             self.norm1 = nn.SyncBatchNorm(dim)
             self.norm2 = nn.SyncBatchNorm(dim)
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)

        self.attn = Attention(dim, lsk_use_3x3=lsk_use_3x3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        ls1 = self.layer_scale_1.view(1, -1, 1, 1)
        ls2 = self.layer_scale_2.view(1, -1, 1, 1)
        x = x + self.drop_path(ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(ls2 * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg and norm_cfg.get('type') == 'SyncBN':
            self.norm = nn.SyncBatchNorm(embed_dim)
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W

@register()
class LSKNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(LayerNorm2d, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, return_layers=None,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None,
                 lsk_use_3x3=False):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.return_layers = return_layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        if hasattr(norm_layer, 'func') and norm_layer.func == nn.LayerNorm:
             norm_layer = partial(LayerNorm2d, eps=1e-6)

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg, lsk_use_3x3=lsk_use_3x3)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

        if init_cfg and init_cfg.get('type') == 'Pretrained' and init_cfg.get('checkpoint'):
            self.load_pretrained(init_cfg['checkpoint'])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, LayerNorm2d):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def load_pretrained(self, checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v

        model_state_dict = self.state_dict()
        for k, v in new_state_dict.items():
            if k in model_state_dict:
                if v.shape != model_state_dict[k].shape:
                    print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
                    continue

        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights with msg: {msg}")

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)
            if self.return_layers:
                if i + 1 in self.return_layers:
                    outs.append(x)
            else:
                outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x
