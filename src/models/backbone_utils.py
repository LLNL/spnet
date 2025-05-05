from torch import nn
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, LastLevelP6P7
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
##
import copy
import torch
import loralib as lora

# FPN
class SoliderSwinSPBackbone(nn.Module):
    def __init__(self, swin, name, use_classifier=False, freeze_backbone=False,
            fpn_dim=256, **kwargs):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        ###
        self.swin = swin
        self.return_layers = {
            1: 'feat_res3',
            2: 'feat_res4',
            3: 'feat_res5',
        }
        self.body = self.get_layers
        self.out_channels = fpn_dim
        if name == 'swin_t':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_s':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_b':
            in_channels_list = [256, 512, 1024]
        else:
            raise Exception
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        if self.use_classifier:
            self.norm = swin.norm
            self.head = swin.head

    def get_layers(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        x, hw_shape = self.swin.patch_embed(x)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        layer_dict = OrderedDict()
        for i, stage in enumerate(self.swin.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i in self.return_layers:
                feat_res_name = self.return_layers[i]
                norm_layer = getattr(self.swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                layer_dict[feat_res_name] = out
        #for k,v in layer_dict.items():
        #    print(k, v.shape)
        #exit()
        return layer_dict

    def _forward(self, x):
        x = self.norm(x)
        x = self.head(x)
        return x

    def forward(self, x, shortcut=False):
        # using the forward method from nn.Sequential
        with torch.set_grad_enabled(not self.freeze_backbone):
            y = self.body(x)
        if self.use_classifier:
            l = self._forward(y['feat_res5'].permute(0, 2, 3, 1))
            if shortcut:
                return l
        z = self.fpn(y)
        if self.use_classifier:
            return z, l
        else:
            return z

# FPN
class SwinSPBackbone(nn.Module):
    def __init__(self, swin, name, use_classifier=False, freeze_backbone=False,
            fpn_dim=256, **kwargs):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        ###
        return_layers = {
            '3': 'feat_res3',
            '5': 'feat_res4',
            '7': 'feat_res5',
        }
        self.body = IntermediateLayerGetter(swin.features, return_layers=return_layers)
        self.out_channels = fpn_dim
        if name == 'swin_t':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_s':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_b':
            in_channels_list = [256, 512, 1024]
        elif name == 'swin_v2_t':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_v2_s':
            in_channels_list = [192, 384, 768]
        elif name == 'swin_v2_b':
            in_channels_list = [256, 512, 1024]
        else:
            raise Exception
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        if self.use_classifier:
            self.norm = swin.norm
            self.permute = swin.permute
            self.avgpool = swin.avgpool
            self.flatten = swin.flatten
            self.head = swin.head

    def _forward(self, x):
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x

    def forward(self, x, shortcut=False):
        # using the forward method from nn.Sequential
        with torch.set_grad_enabled(not self.freeze_backbone):
            y = self.body(x)
            for k,v in y.items():
                y[k] = v.permute(0, 3, 1, 2)
        if self.use_classifier:
            l = self._forward(y['feat_res5'].permute(0, 2, 3, 1))
            if shortcut:
                return l
        z = self.fpn(y)
        if self.use_classifier:
            return z, l
        else:
            return z

### XXX
def replace_layers(model, r=1):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, r=r)
            
        target_attr = module
        if False:#isinstance(module, torch.nn.Conv2d):
            ## simple module
            #print('replaced Conv2d: ', name)
            target_weight = copy.deepcopy(target_attr.weight)
            target_bias = copy.deepcopy(target_attr.bias)
            setattr(model, name, lora.Conv2d(target_attr.in_channels, target_attr.out_channels, kernel_size=target_attr.kernel_size[0], stride=target_attr.stride[0], padding=target_attr.padding[0], dilation=target_attr.dilation, groups=target_attr.groups, bias=target_attr.bias is not None, r=r))
            new_module = getattr(model, name)
            new_module.weight = target_weight
            new_module.bias = target_bias
            new_module.requires_grad_(True)
        elif isinstance(module, torch.nn.Linear):
            #print('replaced Linear: ', name)
            target_weight = copy.deepcopy(target_attr.weight)
            target_bias = copy.deepcopy(target_attr.bias)
            setattr(model, name, lora.Linear(target_attr.in_features, target_attr.out_features, r=r))
            new_module = getattr(model, name)
            new_module.weight = target_weight
            new_module.bias = target_bias
            new_module.requires_grad_(True)
### XXX
def lora_to_orig(model, r=1):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            lora_to_orig(module, r=r)
            
        target_attr = module
        if isinstance(module, lora.Linear):
            target_lora_weight = (target_attr.lora_B @ target_attr.lora_A) * target_attr.scaling
            #print('replaced Linear: ', name)
            target_weight = copy.deepcopy(target_attr.weight)
            target_weight = nn.Parameter(target_weight + target_lora_weight)
            target_bias = copy.deepcopy(target_attr.bias)
            setattr(model, name, torch.nn.Linear(target_attr.in_features, target_attr.out_features))
            new_module = getattr(model, name)
            new_module.weight = target_weight
            new_module.bias = target_bias
            
### XXX
def do_freeze_non_norm(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            do_freeze_non_norm(module)
            
        if isinstance(module, torch.nn.LayerNorm):
            module.requires_grad_(True)
### XXX
# FPN
class ConvnextSPBackbone(nn.Module):
    def __init__(self, convnext, name, use_classifier=False, freeze_backbone=False,
            fpn_dim=256, add_frozen=False, use_lora=False, merge_lora=False, freeze_non_norm=False):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        self.add_frozen = add_frozen
        self.use_lora = use_lora
        self.merge_lora = merge_lora
        self.freeze_non_norm = freeze_non_norm
        ###
        return_layers = {
            '3': 'feat_res3',
            '5': 'feat_res4',
            '7': 'feat_res5',
        }
        self.body = IntermediateLayerGetter(convnext.features, return_layers=return_layers)
        # Freeze backbone except for layer norm
        if self.freeze_non_norm:
            self.body.requires_grad_(False)
            do_freeze_non_norm(self.body)
        # Replace body layers with lora layers
        if self.use_lora:
            self.body.requires_grad_(False)
            replace_layers(self.body)
        # Frozen copy of backbone
        if self.add_frozen:
            self.frozen_body = copy.deepcopy(self.body)
            self.frozen_body.requires_grad_(False)
            # zero out the regular body
            #for p in self.body.parameters():
            #    p.requires_grad_(False)
            #    p.zero_()
            #    p.requires_grad_(True)
        #
        self.out_channels = fpn_dim
        if name == 'convnext_tiny':
            in_channels_list = [192, 384, 768]
        elif name == 'convnext_small':
            in_channels_list = [192, 384, 768]
        elif name == 'convnext_base':
            in_channels_list = [256, 512, 1024]
        elif name == 'convnext_large':
            in_channels_list = [384, 768, 1536]
        else:
            raise Exception
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        if self.use_classifier:
            self.avgpool = convnext.avgpool
            self.classifier = convnext.classifier

    def forward(self, x, shortcut=False):
        # run body
        with torch.set_grad_enabled(not self.freeze_backbone):
            y = self.body(x)
        # add frozen features
        if self.add_frozen:
            y_frozen = self.frozen_body(x)
            for k in y:
                y[k] = y[k] + y_frozen[k]
        # run classifier layer
        if self.use_classifier:
            p = self.avgpool(y['feat_res5'])
            l = self.classifier(p)
            if shortcut:
                return l
        # run FPN
        z = self.fpn(y)
        # return depends on classifier
        if self.use_classifier:
            return z, l
        else:
            return z
            
# ResNet FPN
class ResnetSPBackbone(nn.Module):
    def __init__(self, resnet, name,
            use_classifier=False, freeze_backbone=False,
            fpn_dim=256):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        ###
        return_layers = {
            'layer2': 'feat_res3',
            'layer3': 'feat_res4',
            'layer4': 'feat_res5',
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.out_channels = fpn_dim
        if name == 'resnet50':
            in_channels_list = [512, 1024, 2048]
        else:
            raise Exception
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        if self.use_classifier:
            self.avgpool = resnet.avgpool
            self.classifier = resnet.fc

    def forward(self, x, shortcut=False):
        # using the forward method from nn.Sequential
        with torch.set_grad_enabled(not self.freeze_backbone):
            y = self.body(x)
        if self.use_classifier:
            p = self.avgpool(y['feat_res5'])
            p = torch.flatten(p, 1)
            l = self.classifier(p)
            if shortcut:
                return l
        z = self.fpn(y)
        if self.use_classifier:
            return z, l
        else:
            return z

# ViT FPN
class VitSPBackbone(nn.Module):
    def __init__(self, vit, name,
            use_classifier=False, freeze_backbone=False,
            fpn_dim=256):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        ###
        return_layers = {
            '7': 'feat_res3',
            '9': 'feat_res4',
            '11': 'feat_res5',
        }
        self.vit = vit
        self.patch_embed = self.vit.patch_embed
        self._pos_embed = self.vit._pos_embed
        self.patch_drop = self.vit.patch_drop
        self.norm_pre = self.vit.norm_pre
        self.body = IntermediateLayerGetter(self.vit.blocks, return_layers=return_layers)
        self.out_channels = fpn_dim
        if name == 'vit-s16':
            in_channels_list = [384, 384, 384]
        else:
            raise Exception
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        if self.use_classifier:
            self.norm = resnet.norm
            self.head = resnet.head
        ###
        self.size_dict = {
            'feat_res3': 64,
            'feat_res4': 32,
            'feat_res5': 16,
        }
        ###

    def forward(self, x, shortcut=False):
        # using the forward method from nn.Sequential
        with torch.set_grad_enabled(not self.freeze_backbone):
            ###
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            ###
            y = self.body(x)
            for k, v in y.items():
                y[k] = F.interpolate(v.permute(0, 2, 1)[:, :, :1024].reshape(-1, 384, 32, 32), size=self.size_dict[k])
        if self.use_classifier:
            p = self.norm(y['feat_res5'])
            #p = torch.flatten(p, 1)
            l = self.head(p)
            if shortcut:
                return l
        z = self.fpn(y)
        if self.use_classifier:
            return z, l
        else:
            return z

# PASS ViT FPN
class PassVitSPBackbone(nn.Module):
    def __init__(self, vit, name,
            use_classifier=False, freeze_backbone=False,
            fpn_dim=256):
        super().__init__()
        self.use_classifier = use_classifier
        self.freeze_backbone = freeze_backbone
        ###
        return_layers = {
            '7': 'feat_res3',
            '9': 'feat_res4',
            '11': 'feat_res5',
        }
        self.vit = vit
        self.body = IntermediateLayerGetter(self.vit.blocks, return_layers=return_layers)
        self.out_channels = fpn_dim
        in_channels_list = [self.out_channels, self.out_channels, self.out_channels]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelP6P7(in_channels_list[-1], fpn_dim),
        )
        ###
        self.size_dict = {
            'feat_res3': 64,
            'feat_res4': 32,
            'feat_res5': 16,
        }
        ###

    def forward(self, x, shortcut=False):
        # using the forward method from nn.Sequential
        with torch.set_grad_enabled(not self.freeze_backbone):
            ###
            x = self.vit.prepare_tokens(x, 't', 0)
            ###
            y = self.body(x)
            for k, v in y.items():
                y[k] = F.interpolate(v.permute(0, 2, 1)[:, :, :1024].reshape(-1, self.out_channels, 32, 32), size=self.size_dict[k])
                cls_token = v[:, 1024, :]
        if self.use_classifier:
            l = self.vit.head(cls_token)
            if shortcut:
                return l
        z = self.fpn(y)
        if self.use_classifier:
            return z, l
        else:
            return z
