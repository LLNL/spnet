# Global imports
import copy
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
## Types
from typing import Dict
from torch import Tensor


# Legacy resnet50 backbone
class OldBackbone(nn.Sequential):
    def __init__(self, resnet):
        super(OldBackbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(OldBackbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


# Legacy resnet50 head
class OldRes5Head(nn.Sequential):
    def __init__(self, resnet):
        super(OldRes5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(OldRes5Head, self).forward(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Generic Backbone
class Backbone(nn.Module):
    def forward(self, x):
        y = self.body(x)
        return y


# Generic Head
class Head(nn.Module):
    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}


# Resnet Backbone
class ResnetBackbone(Backbone):
    def __init__(self, resnet):
        super().__init__()
        return_layers = {
            'layer3': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.out_channels = 1024


# Resnet Head
class ResnetHead(Head):
    def __init__(self, resnet):
        super().__init__()
        self.head = resnet.layer4
        self.out_channels = [1024, 2048]
        self.featmap_names = ['feat_res4', 'feat_res5']


# Convnext Backbone
class ConvnextBackbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '5': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(convnext.features, return_layers=return_layers)
        self.out_channels = convnext.features[5][-1].block[5].out_features


# Convnext Head
class ConvnextHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.features[6],
            convnext.features[7],
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,
            convnext.features[7][-1].block[5].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']

# Swin Head
class SwinHead(Head):
    def __init__(self, swin):
        super().__init__()
        self.head = nn.Sequential(
            swin.features[6],
            swin.features[7],
        )
        self.out_channels = [
            swin.features[5][-1].mlp[3].out_features,
            swin.features[7][-1].mlp[3].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']

    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x.permute(0, 2, 3, 1))
        feat = feat.permute(0, 3, 1, 2)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}

# Swin Head
class VitHead(Head):
    def __init__(self, vit, dim):
        super().__init__()
        self.dim = dim
        self.head = nn.Sequential(
            vit.blocks[10],
            vit.blocks[11],
        )
        self.out_channels = [
            self.dim,
            self.dim,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']

    def forward(self, x) -> Dict[str, Tensor]:
        if x.shape[0] > 0:
            feat = self.head(F.interpolate(x, (32, 32)).reshape(-1, self.dim, 1024).permute(0, 2, 1))
            feat = feat.permute(0, 2, 1).reshape(-1, self.dim, 32, 32)
        else:
            feat = x
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}

# resnet model builder function
def build_resnet(arch='resnet50', pretrained=True,
        freeze_backbone_batchnorm=True, freeze_layer1=True, freeze_backbone=False,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    # weights
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    # load model
    if freeze_backbone_batchnorm:
        resnet = torchvision.models.resnet50(weights=weights, norm_layer=norm_layer)
    else:
        resnet = torchvision.models.resnet50(weights=weights)

    # freeze first layers
    resnet.conv1.requires_grad_(False)
    resnet.bn1.requires_grad_(False)
    if freeze_layer1:
        resnet.layer1.requires_grad_(False)

    # setup backbone architecture
    backbone, head = ResnetBackbone(resnet), ResnetHead(resnet)
    
    # freeze whole backbone
    if freeze_backbone:
        backbone.requires_grad_(False)

    # return backbone, head
    return backbone, head


# convnext model builder function
def build_convnext(arch='convnext_base', pretrained=True,
        freeze_layer1=True, freeze_backbone=False):
    # weights
    weights = None

    # load model
    if arch == 'convnext_tiny':
        print('==> Backbone: ConvNext Tiny')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_tiny(weights=weights)
    elif arch == 'convnext_small':
        print('==> Backbone: ConvNext Small')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_small(weights=weights)
    elif arch == 'convnext_base':
        print('==> Backbone: ConvNext Base')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_base(weights=weights)
    elif arch == 'convnext_large':
        print('==> Backbone: ConvNext Large')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_large(weights=weights)
    else:
        raise NotImplementedError

    # freeze first layer
    if freeze_layer1:
        convnext.features[0].requires_grad_(False)

    # setup backbone architecture
    backbone, head = ConvnextBackbone(convnext), ConvnextHead(convnext)
    
    # freeze whole backbone
    if freeze_backbone:
        backbone.requires_grad_(False)

    # return backbone, head
    return backbone, head

### SOLIDER models

class SoliderSwinBackbone(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()
        self.swin = swin
        self.out_channels = out_channels

    def forward(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        x, hw_shape = self.swin.patch_embed(x)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum-1:
                norm_layer = getattr(self.swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return OrderedDict([["feat_res4", outs[-1]]])

class SoliderSwinHead(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()  # last block
        self.swin = swin
        self.out_channels = [out_channels, out_channels*2]
        self.featmap_names = ['feat_res4', 'feat_res5']

    def forward(self, x, bonenum=3):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        x,hw_shape = self.swin.stages[bonenum-1].downsample(x,hw_shape)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == len(self.swin.stages) - bonenum - 1:
                norm_layer = getattr(self.swin, f'norm{bonenum+i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[bonenum+i]).permute(0, 3, 1,
                                                             2).contiguous()
        feat = self.swin.avgpool(feat)
        out = self.swin.avgpool(out)
        return OrderedDict([["feat_res4", feat], ["feat_res5", out]])

### LUPerson models


