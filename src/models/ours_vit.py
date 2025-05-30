
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch._six import container_abcs
import collections.abc as container_abcs


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        #  self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
    
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        #  x = self.norm2(x.transpose(1,2)).transpose(1,2)
        #  x = x + self.drop_path(self.mlp(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out   

class PatchEmbed_VOLO(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          padding=3, bias=False),  # 112x112
                #  nn.BatchNorm2d(hidden_dim),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                #  nn.BatchNorm2d(hidden_dim),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        x = x.flatten(2).permute(0,2,1)
        return x


class Ours_ViT(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.img_size = to_2tuple(img_size)
        self.patch_embed = PatchEmbed_VOLO(img_size=img_size,stem_conv=True, stem_stride=2
                , patch_size=patch_size,in_chans=in_chans, hidden_dim=64,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part1_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part2_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part3_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.in_planes = embed_dim
        # Initialize SIE Embedding

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.part1_pos, std=.02)
        trunc_normal_(self.part2_pos, std=.02)
        trunc_normal_(self.part3_pos, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.part_token1, std=.02)
        trunc_normal_(self.part_token2, std=.02)
        trunc_normal_(self.part_token3, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def interpolate_pos_encoding(self, npatch, dim, h, w):
        N = self.pos_embed.shape[1]
        if npatch == N and self.img_size == (h,w):
            return self.pos_embed
        
        patch_pos_embed = self.pos_embed
        OH = self.img_size[0] // self.patch_embed.patch_size
        OW = self.img_size[1] // self.patch_embed.patch_size
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, OH, OW, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / OH, w0 / OW),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

        
    def prepare_tokens(self, x, model_type, part_index):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        if model_type == 't':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            part_tokens1 = self.part_token1.expand(B, -1, -1)
            part_tokens2 = self.part_token2.expand(B, -1, -1)
            part_tokens3 = self.part_token3.expand(B, -1, -1)
            x = torch.cat((cls_tokens, part_tokens1, part_tokens2, part_tokens3, x), dim=1)
    
            # add positional encoding to each token
            x = x + torch.cat((self.cls_pos, self.part1_pos, self.part2_pos, self.part3_pos, self.interpolate_pos_encoding(x.shape[1], x.shape[-1], w, h)), dim=1)
        else:
            if part_index==0:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                part_tokens1 = self.part_token1.expand(B, -1, -1)
                x = torch.cat((cls_tokens, part_tokens1, x), dim=1)
        
                # add positional encoding to each token
                x = x + torch.cat((self.cls_pos, self.part1_pos, self.interpolate_pos_encoding(x.shape[1], x.shape[-1], w, h)), dim=1)
            elif part_index==1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                part_tokens2 = self.part_token2.expand(B, -1, -1)
                x = torch.cat((cls_tokens, part_tokens2, x), dim=1)
        
                # add positional encoding to each token
                x = x + torch.cat((self.cls_pos, self.part2_pos, self.interpolate_pos_encoding(x.shape[1], x.shape[-1], w, h)), dim=1)
            elif part_index==2:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                part_tokens3 = self.part_token3.expand(B, -1, -1)
                x = torch.cat((cls_tokens, part_tokens3, x), dim=1)
        
                # add positional encoding to each token
                x = x + torch.cat((self.cls_pos, self.part3_pos, self.interpolate_pos_encoding(x.shape[1], x.shape[-1], w, h)), dim=1)

        return self.pos_drop(x) 


    
    def forward(self, x, model_type, part_index):
        x = self.prepare_tokens(x, model_type, part_index)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if model_type=='t':
            return x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        else:
            return x[:, 0], x[:, 1],
            
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x, 't', None)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                #v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
                #v = resize_pos_embed(v, self.pos_embed, 32, 32)
            try:
                self.state_dict()[k].copy_(v)
                count = count + 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))

    @torch.no_grad()
    def _load_weights(self, checkpoint_path: str, prefix: str = '') -> None:
        model = self
        """ Load weights from .npz checkpoints for official Google Brain Flax implementation
        """
        import numpy as np
        from timm.models._manipulate import adapt_input_conv
        from timm.layers import resample_patch_embed, resample_abs_pos_embed

        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        w = np.load(checkpoint_path)
        interpolation = 'bilinear'
        antialias = False
        big_vision = False
        if not prefix:
            if 'opt/target/embedding/kernel' in w:
                prefix = 'opt/target/'
            elif 'params/embedding/kernel' in w:
                prefix = 'params/'
                big_vision = True
            elif 'params/img/embedding/kernel' in w:
                prefix = 'params/img/'
                big_vision = True

        if hasattr(model.patch_embed, 'backbone'):
            # hybrid
            backbone = model.patch_embed.backbone
            stem_only = not hasattr(backbone, 'stem')
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
            stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
            stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                        for r in range(3):
                            getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                            getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                            getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                            block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                            block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
                            print('==> copied: block, {}, {}'.format(i, j))
            embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        else:
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
        if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
            embed_conv_w = resample_patch_embed(
                embed_conv_w,
                model.patch_embed.proj.weight.shape[-2:],
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        model.patch_embed.proj.weight.copy_(embed_conv_w)
        model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
        if model.cls_token is not None:
            model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
        if big_vision:
            pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
        else:
            pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
        if pos_embed_w.shape != model.pos_embed.shape:
            old_shape = pos_embed_w.shape
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                #new_size=model.patch_embed.grid_size,
                new_size=(32,32),
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        model.pos_embed.copy_(pos_embed_w[:, :1024, :])
        model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
        print('==> copied: norm')
        if (isinstance(model.head, nn.Linear) and
                f'{prefix}head/bias' in w and
                model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
            model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
            model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
            print('==> copied: head')
        # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
        # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
        if False:#model.attn_pool is not None:
            block_prefix = f'{prefix}MAPHead_0/'
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
            model.attn_pool.kv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
            model.attn_pool.kv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
            model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
            model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
            model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            for r in range(2):
                getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
                getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

        mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
        for i, block in enumerate(model.blocks.children()):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))
        print('==> copied: blocks')

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def ours_vit_small(img_size=(256, 128), patch_size=16, stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.0, **kwargs):
    #  kwargs.setdefault('qk_scale', 768 ** -0.5)
    print('ICS')
    model = Ours_ViT(img_size=img_size, patch_size=patch_size, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6,  mlp_ratio=4., qkv_bias=True, drop_path_rate = drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ours_vit_base(img_size=(256, 128), patch_size=16, stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.0, **kwargs):
    print('ICS')
    model = Ours_ViT(img_size=img_size, patch_size=patch_size, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12,  mlp_ratio=4., qkv_bias=True, drop_path_rate = drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
