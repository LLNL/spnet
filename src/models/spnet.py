# Global imports
import os
import copy
import math
import scipy.special
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import einops
import timm
## torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
## torchvision
import torchvision
from torchvision.ops import boxes as box_ops, sigmoid_focal_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.utils import _log_api_usage_once
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection._utils import generalized_box_iou_loss
## pytorch metric learning
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner

# Package imports
## Models
from osr.models.backbone_utils import ConvnextSPBackbone, ResnetSPBackbone, SwinSPBackbone, SoliderSwinSPBackbone, lora_to_orig, VitSPBackbone, PassVitSPBackbone
### (requires installation of "fastreid")
try:
    from osr.models.lup_resnet import build_resnet_backbone as lup_build_resnet_backbone
except ModuleNotFoundError:
    pass
from osr.models.backbone import build_resnet, build_convnext, ConvnextHead, SwinHead, SoliderSwinHead, ResnetHead, VitHead
from osr.models.transform import GeneralizedRCNNTransform
from osr.models.swin_transformer import swin_tiny_patch4_window7_224,swin_small_patch4_window7_224,swin_base_patch4_window7_224
### GFN
from osr.models.gfn import GalleryFilterNetwork, SafeBatchNorm1d
from osr.models.box_ops import box_iou_ew
### PASS
from osr.models.ours_vit import ours_vit_small as PASSViT_small
from osr.models.ours_vit import ours_vit_base as PASSViT_base
## Losses
from osr.losses.oim_loss import OIMLossSafe, OIMLossCQ


### Helper function
def _copy_key_prefix(state_dict, prefix_src, prefix_dst):
    key_list = list(state_dict.keys())
    for key_src in key_list:
        for key_dst in key_list:
            if key_src.startswith(prefix_src) and key_dst.startswith(prefix_dst):
                suffix_src = key_src.split(prefix_src)[1]
                suffix_dst = key_dst.split(prefix_dst)[1]
                if suffix_src == suffix_dst:
                    state_dict[key_dst] = state_dict[key_src]

### Randomly sample indices from binary mask
def _subsample_mask(mask, num):
    i1, i2 = torch.where(mask)
    if len(i1) > num:
        rand_idx = torch.randperm(len(i1))[:num]
        i1, i2 = i1[rand_idx], i2[rand_idx]
    return i1, i2

class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float,
            use_all_pos=False, neg_per_pos=1) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_all_pos = use_all_pos
        self.neg_per_pos = neg_per_pos

    def __call__(self, matched_idxs: List[Tensor], scores=None) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched_idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        mask_list = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 0)[0]
            negative = torch.where(matched_idxs_per_image == -1)[0]

            if self.use_all_pos:
                num_pos = positive.numel()
                num_neg = num_pos * self.neg_per_pos
                num_neg = min(negative.numel(), num_neg)
            else:
                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)
                num_neg = self.batch_size_per_image - num_pos
                # protect against not enough negative examples
                num_neg = min(negative.numel(), num_neg)
                #assert (num_pos + num_neg) == self.batch_size_per_image

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.bool)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.bool)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            per_image_mask = pos_idx_per_image_mask | neg_idx_per_image_mask

            mask_list.append(per_image_mask)

        split_lens = [mask.sum().item() for mask in mask_list]
        mask = torch.cat(mask_list)

        return mask, split_lens

class NormAwareEmbedding(nn.Module):
    def __init__(self, featmap_names=["feat_res4", "feat_res5"],
            in_channels=[1024, 2048], dim=256, norm_type='batchnorm'):
        super().__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        if norm_type == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_type == 'batchnorm':
            norm_layer = SafeBatchNorm1d
        elif norm_type == 'groupnorm':
            norm_layer = nn.GroupNorm

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            if norm_type == 'none':
                proj = nn.Linear(in_channel, indv_dim)
                nn.init.normal_(proj.weight, std=0.01)
                nn.init.constant_(proj.bias, 0)
            else:
                if norm_type == 'batchnorm':
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim),
                        norm_layer(indv_dim, affine=False))
                elif norm_type == 'groupnorm':
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim), norm_layer(num_groups=32, num_channels=indv_dim))
                else:
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim), norm_layer(indv_dim))
                nn.init.normal_(proj[0].weight, std=0.01)
                nn.init.constant_(proj[0].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps: Dict[str, Tensor]):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        embeddings = torch.empty(0)
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            fk, fv = list(featmaps.items())[0]
            fv = self._flatten_fc_input(fv)
            ## Loop for torch.jit
            for pk, pv in self.projectors.items():
                if pk == fk:
                    embeddings = pv(fv)
            return embeddings
        else:
            outputs = []
            for fk, fv in featmaps.items():
                fv = self._flatten_fc_input(fv)
                ## Loop for torch.jit
                for pk, pv in self.projectors.items():
                    if pk == fk:
                        outputs.append(pv(fv))
            embeddings = torch.cat(outputs, dim=1)
            return embeddings

    def _flatten_fc_input(self, x):
        if len(x.shape) == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

def _box_loss(
    type,
    box_coder,
    anchors_per_image,
    matched_gt_boxes_per_image,
    bbox_regression_per_image,
    cnf=None,
    reduction='none',
):
    torch._assert(type in ["l1", "smooth_l1", "ciou", "diou", "giou", "mgiou"], f"Unsupported loss: {type}")

    if type == "l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        return F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
    elif type == "smooth_l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
        return F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum", beta=beta)
    else:
        bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        eps = cnf["eps"] if cnf is not None and "eps" in cnf else 1e-7
        if type == "giou":
            return generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction=reduction, eps=eps)
        if type == "mgiou":
            giou_loss = generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, eps=eps)
            mgiou_loss = torch.relu(giou_loss - 0.5).sum()
            return mgiou_loss

def _basic_anchorgen():
    anchor_sizes = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios,
    )
    return anchor_generator

def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator

def _recover_feats1(emb, shape_list, num_anchors):
    # emb in (N, LHWA, D)
    # shape_list in L * [N, H, W, A]
    A = num_anchors
    i = 0
    features = []
    for H, W in shape_list:
        HWA = H * W * A
        # N, HWA, D
        emb_l = emb[:, i:i+HWA, :]
        i += HWA
        emb_l = einops.rearrange(emb_l, 'N (H W A) D -> (N A) D H W', H=H, W=W)
        # NA, D, H, W
        features.append(emb_l)
    return features

def _recover_feats2(emb, shape_list, num_anchors):
    # emb in (N, LHWA, D)
    # shape_list in L * [N, H, W, A]
    NA, _, D = emb.shape
    A = num_anchors
    N = NA // A
    i = 0
    features = []
    _features = []
    for H, W in shape_list:
        HW = H * W 
        # NA, HW, D
        emb_l = emb[:, i:i+HW, :]
        i += HW
        _features.append(einops.rearrange(emb_l, '(N A) (H W) D -> N (H W A) D', N=N, A=A, H=H, W=W, D=D))
        features.append(einops.rearrange(emb_l, '(N A) (H W) D -> (N A) D H W', N=N, A=A, H=H, W=W, D=D))
    _features = torch.cat(_features, dim=1)
    return _features, features

class AlignEmbHead(nn.Module):
    def __init__(self, emb_head, nae_head):
        super().__init__()
        self.emb_head = emb_head
        self.nae_head = nae_head

    def forward(self, x, return_feat=False):
        y = self.emb_head(x)
        z = self.nae_head(y)
        if return_feat:
            return z, y['feat_res5']
        else:
            return z

class DifferenceCombiner(nn.Module):
    def __init__(self, scaler=100.0):
        super().__init__()
        self.scaler = scaler

    def norm(self, q, a, m=None):
        if self.training:
            if m is not None:
                a_idx, q_idx = torch.where(m)
                _q = q[q_idx]
                _a = a[a_idx]
                norm = (_q.unsqueeze(1) - _a).norm(dim=2)
            else:
                x = torch.einsum('qk,qk->q',
                    q / self.scaler, q / self.scaler).view(1, -1, 1)
                y = torch.einsum('nak,nak->na',
                    a / self.scaler, a / self.scaler).unsqueeze(1)
                xy = torch.einsum('qk,nak->nqa',
                    q / self.scaler, a / self.scaler)
                norm = torch.sqrt(x + y - 2*xy) * self.scaler
        else:
            x = torch.einsum('ik,ik->i', q, q).unsqueeze(1)
            y = torch.einsum('jk,jk->j', a, a).unsqueeze(0)
            xy = torch.einsum('ik,jk->ij', q, a)
            norm = torch.sqrt(x + y - 2*xy)
        return norm

    def forward(self, q, a):
        o = q - a
        return o 

class ProductCombiner(nn.Module):
    def __init__(self, t=0.1, e=1e-5, scaler=100.0):
        super().__init__()
        self.t = t
        self.e = e
        self.scaler = scaler

    def norm_prod_new(self, q, a):
        if self.training:
            x = q ** 2
            y = torch.sigmoid(-a / self.t) ** 2
            z = torch.einsum('ik,njk->nij', x, y)
            norm = torch.sqrt(z)
        else:
            x = q ** 2
            y = torch.sigmoid(-a / self.t) ** 2
            z = torch.einsum('ik,jk->ij', x, y)
            norm = torch.sqrt(z)
        return norm

    def norm_prod(self, q, a):
        if self.training:
            x = torch.sigmoid(-q / self.t) ** 2
            y = a ** 2
            z = torch.einsum('ik,njk->nij', x, y)
            norm = torch.sqrt(z)
        else:
            x = torch.sigmoid(-q / self.t) ** 2
            y = a ** 2
            z = torch.einsum('ik,jk->ij', x, y)
            norm = torch.sqrt(z)
        return norm

    def norm_prod_old(self, q, a):
        if self.training:
            x = torch.sigmoid(q / self.t) ** 2
            y = a ** 2
            z = torch.einsum('ik,njk->nij', x, y)
            norm = torch.sqrt(z)
        else:
            x = torch.sigmoid(q / self.t) ** 2
            y = a ** 2
            z = torch.einsum('ik,jk->ij', x, y)
            norm = torch.sqrt(z)
        return norm

    def norm(self, q, a, m=None):
        if self.training:
            if m is not None:
                a_idx, q_idx = torch.where(m)
                _q = q[q_idx]
                _a = a[a_idx]
                norm = (_q.unsqueeze(1) - _a).norm(dim=2)
            else:
                x = torch.einsum('qk,qk->q',
                    q / self.scaler, q / self.scaler).view(1, -1, 1)
                y = torch.einsum('nak,nak->na',
                    a / self.scaler, a / self.scaler).unsqueeze(1)
                xy = torch.einsum('qk,nak->nqa',
                    q / self.scaler, a / self.scaler)
                norm = torch.sqrt(x + y - 2*xy) * self.scaler
        else:
            x = torch.einsum('ik,ik->i', q, q).unsqueeze(1)
            y = torch.einsum('jk,jk->j', a, a).unsqueeze(0)
            xy = torch.einsum('ik,jk->ij', q, a)
            norm = torch.sqrt(x + y - 2*xy)
        return norm

    def forward(self, q, a):
        o = a * torch.sigmoid(q / self.t)
        return o 

class BridgeLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        # Init layer to identity
        #nn.init.eye_(self.linear.weight)
        #nn.init.zeros_(self.linear.bias)
        # Init layer to zeros
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Alt param
        #nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
        #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, a):
        q = self.linear(a)
        o = q - a 
        return o

class SPNetHead(nn.Module):
    """
    A regression and classification head for use in SPNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    FG = 0
    BG = -1
    NM = -2

    def __init__(self, config, in_channels, num_anchors, num_classes, norm_layer: Optional[Callable[..., nn.Module]] = None, oim_lut_size=None, emb_head=None):
        super().__init__()
        self.match_mode = config['match_mode']
        self.match_conservative = config['match_conservative']
        self.train_mode = config['train_mode'] 
        self.test_mode = config['test_mode'] 
        self.emb_align_mode = config['emb_align_mode']
        self.emb_align_sep = config['emb_align_sep']
        self.use_moco = config['use_moco']
        self.moco_copy_teacher = config['moco_copy_teacher']
        self.num_anchors = num_anchors
        self.num_cascade_steps = config['num_cascade_steps']
        self.emb_loc_dim = config['emb_dim']
        self.emb_reid_dim = config['emb_reid_dim']
        self.reid_loss_weight = config['reid_loss_weight']
        self.use_gfn = config['use_gfn']
        self.reid_only = config['reid_only']
        self.use_anchor_head = config['use_anchor_head']
        self.use_posnorm = config['use_posnorm']

        # Subsampler
        self.subsample_per_query = config['subsample_per_query']
        self.subsample_per_image = config['subsample_per_image']
        self.subsample_positive_fraction = 0.5
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            self.subsample_per_image, self.subsample_positive_fraction,
            neg_per_pos=config['subsample_neg_per_pos'])

        # Set head modules (anchor, class, box)
        self.anchor_classification_head = SPNetAnchorClassificationHead(
            config['emb_dim'],
            logits=config['anchor_logits'],
            focal_alpha=config['anchor_focal_alpha'],
            focal_gamma=config['anchor_focal_gamma'],
            pos_smoothing=config['anchor_pos_smoothing'],
            use_posnorm=config['use_posnorm'])
        self.classification_head = nn.ModuleDict()
        self.regression_head = nn.ModuleDict()
        self.bridge_layer = nn.ModuleDict()
        for i in range(config['num_cascade_steps']+1):
            self.classification_head[str(i)] = SPNetClassificationHead(
                config['emb_dim'], num_classes,
                focal_alpha=config['class_focal_alpha'], 
                focal_gamma=config['class_focal_gamma'],
                logits=config['cls_logits'],
                loss_func=config['cls_loss_func'],
                pos_smoothing=config['cls_pos_smoothing'],
                use_posnorm=config['use_posnorm'])
            self.regression_head[str(i)] = SPNetRegressionHead(
                config['emb_dim'])
            if config['bridge_type'] == 'direct':
                self.bridge_layer[str(i)] = nn.Linear(config['emb_dim'], config['emb_dim'])
                nn.init.eye_(self.bridge_layer[str(i)].weight)
                self.bridge_layer[str(i)].weight.data = -self.bridge_layer[str(i)].weight.data
                nn.init.zeros_(self.bridge_layer[str(i)].bias)
            elif config['bridge_type'] == 'combiner':
                self.bridge_layer[str(i)] = BridgeLayer(config['emb_dim'])

        self.feature_head = SPNetFeatureHead(
            in_channels, num_anchors, config['emb_dim'], norm_layer=norm_layer)
        if self.emb_align_mode == 'conv5':
            self.featmap_names = ['feat_res3', 'feat_res4', 'feat_res5', 'p6', 'p7']
            self.roi_align = MultiScaleRoIAlign(
                featmap_names=[config['featmap_name']],
                output_size=config['featmap_size'], sampling_ratio=2
            )
            nae_head = NormAwareEmbedding(
                    featmap_names=emb_head.featmap_names,
                    in_channels=emb_head.out_channels,
                    dim=self.emb_reid_dim,
                    norm_type=config['emb_norm_type'],
                )
            self.align_emb_head = AlignEmbHead(
                emb_head,
                nae_head
            )
            # Align sep case
            if self.emb_align_sep:
                class LocHead(nn.Module):
                    def __init__(self, emb_head, emb_dim, feat_dim):
                        super().__init__()
                        self.emb_head = emb_head
                        self.head = nn.Linear(feat_dim, emb_dim, bias=False)

                    def forward(self, x):
                        x = self.emb_head(x)
                        x = self.head(x['feat_res5'].flatten(start_dim=1))
                        return x

                if config['backbone_arch'] in ('convnext_tiny', 'swin_t', 'swin_s', 'swin_v2_t', 'swin_v2_s'):
                    feat_res5_dim = 768
                elif config['backbone_arch'] in ('convnext_base', 'swin_b', 'swin_v2_b'):
                    feat_res5_dim = 1024
                elif config['backbone_arch'] in ('resnet50',):
                    feat_res5_dim = 2048
                self.loc_featmap_names = ['feat_res3', 'feat_res4', 'feat_res5', 'p6', 'p7']
                self.loc_roi_align = MultiScaleRoIAlign(
                    featmap_names=[config['featmap_name']],
                    output_size=config['featmap_size'], sampling_ratio=2
                )
                self.align_emb_loc_head = nn.ModuleDict()
                if config['num_cascade_steps'] == 0:
                    self.align_emb_loc_head["1"] = LocHead(
                        copy.deepcopy(emb_head), config['emb_dim'],
                        feat_dim=feat_res5_dim,
                    )
                else:
                    for i in range(1, config['num_cascade_steps']+1):
                        self.align_emb_loc_head[str(i)] = LocHead(
                            copy.deepcopy(emb_head), config['emb_dim'],
                            feat_dim=feat_res5_dim,
                        )
        else:
            self.featmap_names = ["0", "1", "2", "3", "4"]
            self.roi_align = MultiScaleRoIAlign(featmap_names=self.featmap_names,
                output_size=3, sampling_ratio=2)
            self.align_emb_head = nn.Sequential(
                nn.Conv2d(in_channels, config['emb_dim'], kernel_size=3, stride=1, padding=0),
            )

        # MOCO
        if self.use_moco or self.moco_copy_teacher:
            self.moco_align_emb_head = copy.deepcopy(self.align_emb_head)
            self.moco_align_emb_head.requires_grad_(False)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.proposal_matcher = det_utils.Matcher(
            0.5,
            0.4,
            allow_low_quality_matches=True,
        )
        self.cascade_matcher = det_utils.Matcher(
            0.5,
            0.5,
            allow_low_quality_matches=False,
        )
        self.reid_objective = config['reid_objective']
        self.reid_gt_only = config['reid_gt_only']
        self.oim_lut_size = oim_lut_size
        if self.reid_objective == 'oim':
            # Only supports object-centric training for now
            assert not config['use_ssl']
            class OIMLossWrapper(nn.Module):
                def __init__(self, emb_dim, oim_lut_size,
                        oim_cq_size, oim_momentum, oim_scalar):
                    super().__init__()
                    self.train_reid_loss = OIMLossSafe(emb_dim, oim_lut_size[0],
                        oim_cq_size, oim_momentum, oim_scalar)
                    self.test_reid_loss = OIMLossSafe(emb_dim, oim_lut_size[1],
                        oim_cq_size, oim_momentum, oim_scalar)

                def forward(self, *args):
                    if self.training:
                        return self.train_reid_loss(*args)
                    else:
                        return self.test_reid_loss(*args)

            if config['test_eval_mode'] == 'loss':
                self.reid_loss = OIMLossWrapper(config['emb_reid_dim'], oim_lut_size,
                    config['oim_cq_size'], config['oim_momentum'], config['oim_scalar'])
            else:
                self.reid_loss = OIMLossSafe(config['emb_reid_dim'], oim_lut_size[0],
                    config['oim_cq_size'], config['oim_momentum'], config['oim_scalar'])
        elif self.reid_objective == 'cq':
            class CQLossWrapper(nn.Module):
                def __init__(self, emb_dim,
                        oim_cq_size, oim_scalar, oim_momentum):
                    super().__init__()
                    self.train_reid_loss = OIMLossCQ(emb_dim,
                        oim_cq_size, oim_scalar, oim_momentum)
                    self.test_reid_loss = OIMLossCQ(emb_dim,
                        oim_cq_size, oim_scalar, oim_momentum)

                def forward(self, *args, **kwargs):
                    if self.training:
                        return self.train_reid_loss(*args, **kwargs)
                    else:
                        return self.test_reid_loss(*args, **kwargs)

            if config['test_eval_mode'] in ('loss', 'all'):
                self.reid_loss = CQLossWrapper(config['emb_reid_dim'], config['oim_cq_size'],
                    config['oim_scalar'], config['oim_momentum'])
            else:
                self.reid_loss = OIMLossCQ(config['emb_reid_dim'], config['oim_cq_size'],
                    config['oim_scalar'], config['oim_momentum'])
        elif self.reid_objective == 'ntx':
            self.reid_miner = BatchEasyHardMiner()
            self.reid_loss = NTXentLoss(temperature=0.1)



        ### COMBINER
        if config['combiner'] == 'diff':
            self.combiner = DifferenceCombiner()
        elif config['combiner'] == 'prod':
            self.combiner = ProductCombiner()

        # Gallery-Filter Network
        if self.use_gfn:
            ## Build Gallery Filter Network
            self.gfn = GalleryFilterNetwork(
                self.roi_align, emb_head, nae_head,
                featmap_name=config['featmap_name'],
                mode=config['gfn_mode'],
                gfn_activation_mode=config['gfn_activation_mode'],
                emb_dim=config['emb_reid_dim'], temp=config['gfn_train_temp'],
                se_temp=config['gfn_se_temp'],
                filter_neg=config['gfn_filter_neg'],
                use_image_lut=config['gfn_use_image_lut'],
                gfn_query_mode=config['gfn_query_mode'],
                gfn_scene_pool_size=config['gfn_scene_pool_size'],
                gfn_norm_type=config['emb_norm_type'],
                pos_num_sample=config['gfn_num_sample'][0],
                neg_num_sample=config['gfn_num_sample'][1],
                reid_loss=self.reid_loss)
        else:
            self.gfn = None

    def read_gt_embeddings(self, targets):
        gt_emb_list = [torch.cat(target['query_loc_emb']) for target in targets]
        return gt_emb_list

    def get_gt_embeddings(self, features, targets, moco=False, emb_type='reid'):
        boxes_list = [t['boxes'] for t in targets]
        image_shapes = [t['image_shape'] for t in targets]

        # get features
        if (emb_type == 'reid') or (not self.emb_align_sep):
            features = dict(zip(self.featmap_names, features))
            gt_feat = self.roi_align(features, boxes_list, image_shapes)
        elif emb_type == 'loc':
            features = dict(zip(self.loc_featmap_names, features))
            gt_feat = self.loc_roi_align(features, boxes_list, image_shapes)
        gt_emb_list = []
        for _gt_feat in torch.split(gt_feat, 8192):
            if moco:
                _gt_emb = self.moco_align_emb_head(_gt_feat)
            else:
                if (emb_type == 'reid') or (not self.emb_align_sep):
                    _gt_emb = self.align_emb_head(_gt_feat)
                elif emb_type == 'loc':
                    _gt_emb = self.align_emb_loc_head["1"](_gt_feat)
            gt_emb_list.append(_gt_emb)
        gt_emb = torch.cat(gt_emb_list)
        N, D, *_ = gt_emb.shape
        gt_emb = gt_emb.reshape(N, D)
        gt_emb_list = torch.split(gt_emb, [len(b) for b in boxes_list])
        return gt_emb_list

    def compute_loss_classifier(self, class_logits, targets, lwf_logits=None):
        if lwf_logits is None:
            labels = torch.cat([t['image_label'] for t in targets])
            loss = F.cross_entropy(class_logits, labels, label_smoothing=0.1)
        else:
            loss = F.binary_cross_entropy_with_logits(
                class_logits.div(2).log_softmax(dim=1),
                lwf_logits.div(2).softmax(dim=1),
                reduction='mean')
        return loss

    def compute_loss_emb_tfm(self, head_outputs, cascade_dict, targets, use_nms=False, nms_thresh=0.99, subsample=True):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        features = dict(zip(self.featmap_names, cascade_dict['features']))
        image_shapes = [target['image_shape'] for target in targets]
        ###

        # Get GT embeddings and labels
        gt_emb = cascade_dict['gt_emb']
        # get gt pids
        if self.reid_objective == 'oim':
            gt_pid = torch.cat([target["labels"] for target in targets]) - 1
        else:
            gt_pid = torch.cat([target["person_id"] for target in targets])
        # MOCO embeddings
        if self.use_moco:
            moco_gt_emb = torch.cat(head_outputs['moco_gt_emb'])

        # decode boxes
        if not self.reid_gt_only:
            boxes = self.box_coder.decode_single(
                cascade_dict['bbox_regression'], cascade_dict['anchors'],
            )

            # clip boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, image_shapes[0]).detach()

            # get mask of boxes with high IoU with target
            match_iou = box_iou_ew(boxes, cascade_dict['box_target'])
            iou_mask = match_iou >= 0.7 # 0.5

            # subsample positives
            if subsample and (cascade_dict['query_split_lens'] is not None):
                ## Sample at most k//4 boxes per image
                new_iou_mask_list = []
                for _iou_mask in iou_mask.split(cascade_dict['query_split_lens']):
                    _iou_idx = torch.where(_iou_mask)[0]
                    _rand_idx = torch.randperm(len(_iou_idx))[:self.subsample_per_image//4]
                    _rand_iou_idx = _iou_idx[_rand_idx]
                    _new_iou_mask = torch.zeros_like(_iou_mask)
                    _new_iou_mask[_rand_iou_idx] = True
                    new_iou_mask_list.append(_new_iou_mask)
                iou_mask = torch.cat(new_iou_mask_list)

            # get embeddings only for boxes with high IoU with target
            split_lens = cascade_dict['split_lens']
            iou_mask_list = torch.split(iou_mask, split_lens)

            # mask boxes
            _boxes_list = torch.split(boxes, split_lens)
            boxes_list = [b[m] for b, m in zip(_boxes_list, iou_mask_list)]

            # perform NMS
            if use_nms:
                scores = cascade_dict['cls_logits'][:, 1]
                _scores_list = torch.split(scores, split_lens)
                scores_list = [s[m] for s, m in zip(_scores_list, iou_mask_list)]
                #
                nms_idx_list = [box_ops.nms(box, scores, nms_thresh) for box, scores in zip(boxes_list, scores_list)]
                #
                boxes_list = [b[i] for b, i in zip(boxes_list, nms_idx_list)]

            # get box features
            box_feat = self.roi_align(features, boxes_list, image_shapes)
            box_emb = self.align_emb_head(box_feat)
            N, D, *_ = box_emb.shape
            box_emb = box_emb.reshape(N, D)

            # MOCO embeddings
            if self.use_moco:
                moco_features = dict(zip(self.featmap_names, head_outputs['moco_features']))
                moco_box_feat = self.roi_align(moco_features, boxes_list, image_shapes)
                moco_box_emb = self.moco_align_emb_head(moco_box_feat)
                moco_box_emb = moco_box_emb.reshape(N, D)

            assert gt_emb.shape[0] == gt_pid.shape[0]

            # get box pids
            if use_nms:
                box_pid = cascade_dict['box_pid']
                _pid_list = torch.split(box_pid, split_lens)
                pid_list = [p[m] for p, m in zip(_pid_list, iou_mask_list)]
                pid_list = [p[i] for p, i in zip(pid_list, nms_idx_list)]
                box_pid = torch.cat(pid_list)
            else:
                box_pid = cascade_dict['box_pid'][iou_mask]

        # Subsample embeddings to reduce memory cost of reid loss
        ## (mainly for SSL scenario with many annotations)
        if self.reid_objective in ('oim', 'cq'):
            ## Use only ground truth boxes and not detected boxes
            if self.reid_gt_only:
                loss_emb = gt_emb
                loss_pid = gt_pid
                ### MOCO embeddings
                if self.use_moco:
                    moco_loss_emb = moco_gt_emb
            ## Use ground truth and detected boxes
            else:
                loss_emb = torch.cat([box_emb, gt_emb])
                loss_pid = torch.cat([box_pid, gt_pid])
                ### MOCO embeddings
                if self.use_moco:
                    moco_loss_emb = torch.cat([moco_box_emb, moco_gt_emb])
        else:
            max_emb = 10000
            num_box_emb = box_emb.shape[0]
            num_gt_emb = gt_emb.shape[0]
            num_emb = max(0, max_emb - num_gt_emb)
            if num_gt_emb > max_emb:
                rand_emb_idx = torch.randperm(num_gt_emb)[:max_emb]
                loss_emb = gt_emb[rand_emb_idx]
                loss_pid = gt_pid[rand_emb_idx]
            else:
                rand_emb_idx = torch.randperm(num_box_emb)[:num_emb]
                rand_box_emb = box_emb[rand_emb_idx]
                rand_box_pid = box_pid[rand_emb_idx]
                #
                loss_emb = torch.cat([rand_box_emb, gt_emb])
                loss_pid = torch.cat([rand_box_pid, gt_pid])

        ## compute loss
        ### mean comparison
        if self.reid_objective in ('oim', 'cq'):
            assert loss_emb.shape[0] == loss_pid.shape[0], '{}, {}'.format(loss_emb.shape[0], loss_pid.shape[0])
            if self.use_moco:
                assert loss_emb.shape == moco_loss_emb.shape
                reid_loss = self.reid_loss(loss_emb, loss_pid, moco_inputs=moco_loss_emb)
            else:
                reid_loss = self.reid_loss(loss_emb, loss_pid)
        else:
            unique_loss_pid = torch.unique(gt_pid)
            unique_loss_emb_list = []
            for pid in unique_loss_pid:
                pid_mask = gt_pid == pid
                _unique_loss_emb = torch.mean(gt_emb[pid_mask], dim=0)
                unique_loss_emb_list.append(_unique_loss_emb)
            unique_loss_emb = torch.stack(unique_loss_emb_list)
            #
            if self.reid_objective == 'ntx':
                with torch.no_grad():
                    miner_output = self.reid_miner(loss_emb, loss_pid,
                        ref_emb=unique_loss_emb, ref_labels=unique_loss_pid)
                reid_loss = self.reid_loss(loss_emb, loss_pid, miner_output,
                    ref_emb=unique_loss_emb, ref_labels=unique_loss_pid)
            elif self.reid_objective == 'cq':
                reid_loss = self.reid_loss(loss_emb, loss_pid,
                    unique_loss_emb, unique_loss_pid)

        # return loss dict
        return reid_loss

    def get_offset_emb(self, cascade_dict, targets, cascade_idx,
            use_nms=False, nms_thresh=0.99, add_gt=True, subsample=True):
        image_shapes = [target['image_shape'] for target in targets]
        ###

        # decode boxes
        boxes = self.box_coder.decode_single(
            cascade_dict['bbox_regression'],
            cascade_dict['anchors'],
        )

        # clip boxes to image
        boxes = box_ops.clip_boxes_to_image(boxes, image_shapes[0]).detach()

        # Default split lens
        split_lens = cascade_dict['split_lens']

        # Add GT props
        query_split_lens = None
        target_emb = None
        if add_gt:
            if self.train_mode == 'oc':
                # Add GT
                new_split_lens = []
                new_boxes, new_scores = [], []
                scores = cascade_dict['cls_logits'][:, 1]
                for box, score, target in zip(
                        boxes.split(split_lens), scores.split(split_lens),
                        targets):
                    new_boxes.append(box)
                    new_boxes.append(target['boxes'])
                    new_scores.append(score)
                    new_scores.append(torch.ones(target['boxes'].shape[0]).to(score.device))
                    new_split_lens.append(len(box)+len(target['boxes']))
                boxes = torch.cat(new_boxes)
                cls_logits = torch.cat(new_scores)
                split_lens = new_split_lens
            elif self.train_mode == 'qc':
                query_split_lens = cascade_dict['query_split_lens']
                #
                topk_matched_idxs = cascade_dict['topk_matched_idxs']
                box_target = cascade_dict['box_target']
                box_pid = cascade_dict['box_pid']
                # Add GT
                new_split_lens = []
                new_query_split_lens = []
                new_boxes, new_scores = [], []
                new_embs = []
                new_box_targets, new_idxs, new_pids = [], [], []
                scores = cascade_dict['cls_logits'][:, 1]
                target_emb = cascade_dict['target_emb']
                box_pid = cascade_dict['box_pid']
                for box, score, box_t, emb, idx, pid in zip(
                        boxes.split(query_split_lens), scores.split(query_split_lens),
                        box_target.split(query_split_lens),
                        target_emb.split(query_split_lens),
                        topk_matched_idxs.split(query_split_lens),
                        box_pid.split(query_split_lens)):
                    new_boxes.append(box)
                    new_boxes.append(box_t[0].unsqueeze(0))
                    new_scores.append(score)
                    new_scores.append(torch.ones(1).to(score.device))
                    new_query_split_lens.append(len(box)+1)
                    new_embs.append(emb)
                    new_embs.append(emb[0].unsqueeze(0))
                    #
                    new_box_targets.append(box_t)
                    new_box_targets.append(box_t[0].unsqueeze(0))
                    new_idxs.append(idx)
                    new_idxs.append(torch.zeros(1).to(idx.device))
                    new_pids.append(pid)
                    new_pids.append(pid.max().unsqueeze(0))
                boxes = torch.cat(new_boxes)
                cls_logits = torch.cat(new_scores)
                target_emb = torch.cat(new_embs)
                box_target = torch.cat(new_box_targets)
                topk_matched_idxs = torch.cat(new_idxs)
                box_pid = torch.cat(new_pids)
                query_split_lens = new_query_split_lens
                split_lens = [int(s*1.01) for s in split_lens]
                assert sum(split_lens) == sum(query_split_lens)
        else:
            cls_logits = cascade_dict['cls_logits'][:, 1]
            if self.train_mode == 'qc':
                topk_matched_idxs = cascade_dict['topk_matched_idxs']
                box_target = cascade_dict['box_target']
                box_pid = cascade_dict['box_pid']
                target_emb = cascade_dict['target_emb']
                query_split_lens = cascade_dict['query_split_lens']

        # NMS
        if use_nms:
            Q = len(targets)
            ## Image index
            image_index = torch.cat([torch.tensor([i]).repeat(l) for i, l in enumerate(split_lens)]).to(boxes.device)
            ##
            nms_idx = box_ops.batched_nms(boxes,
                cls_logits, image_index, nms_thresh)
            ##
            boxes = boxes[nms_idx]
            ##
            keep_image_index = image_index[nms_idx]
            split_sections_dict = dict(zip(range(Q), Q*[0]))
            unique_vals, unique_counts = keep_image_index.unique(return_counts=True)
            _split_sections_dict = dict(zip(unique_vals.tolist(), unique_counts.tolist()))
            split_sections_dict = {**split_sections_dict, **_split_sections_dict}
            split_lens = list(split_sections_dict.values())

        # Rematch samples
        if self.train_mode == 'oc':
            matched_idxs = []
            box_targets = []
            box_pids = []
            
            for anchors_per_image, targets_per_image in zip(boxes.split(split_lens), targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                    )
                    box_targets.append(
                        torch.full((anchors_per_image.size(0), 4), -1, dtype=boxes.dtype, device=anchors_per_image.device)
                    )
                    box_pids.append(
                        torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                    )
                else:

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    new_idx = self.cascade_matcher(match_quality_matrix)
                    _new_idx = new_idx.clone()
                    _new_idx[_new_idx<0] = 0
                    box_targets.append(targets_per_image["boxes"][_new_idx])
                    box_pids.append(targets_per_image["labels"][_new_idx]-1)
                    matched_idxs.append(new_idx)
            
            topk_matched_idxs = torch.cat(matched_idxs)
            box_target = torch.cat(box_targets)
            box_pid = torch.cat(box_pids)
        elif self.train_mode == 'qc':
            iou = box_iou_ew(box_target, boxes)
            new_topk_matched_idxs = torch.full_like(topk_matched_idxs, -1)
            new_topk_matched_idxs[iou >= 0.5] = 0
            #
            topk_matched_idxs = new_topk_matched_idxs

        # Subsample to reduce memory usage
        if self.train_mode == 'oc':
            if subsample:
                keep_mask, split_lens = self.fg_bg_sampler(
                    topk_matched_idxs.split(split_lens), scores=cls_logits)
                #
                boxes = boxes[keep_mask] 
                #
                topk_matched_idxs = topk_matched_idxs[keep_mask]
                box_target = box_target[keep_mask]
                box_pid = box_pid[keep_mask]

        # get embeddings for all matching boxes
        boxes_list = torch.split(boxes, split_lens)
        if self.emb_align_sep:
            features = dict(zip(self.loc_featmap_names, cascade_dict['features']))
            box_feat = self.loc_roi_align(features, boxes_list, image_shapes)
            box_emb = self.align_emb_loc_head[str(cascade_idx+1)](box_feat)
        else:
            features = dict(zip(self.featmap_names, cascade_dict['features']))
            box_feat = self.roi_align(features, boxes_list, image_shapes)
            box_emb = self.align_emb_head(box_feat)
        N, D, *_ = box_emb.shape
        box_emb = box_emb.reshape(N, D)

        # get offset emb
        ## qc mode: get offset emb from target and box emb
        if self.train_mode == 'qc':
            assert target_emb.shape == box_emb.shape
            # compute offset emb
            offset_emb = self.combiner.forward(target_emb, box_emb)
        ## oc mode: predict offset emb from box emb
        elif self.train_mode == 'oc':
            offset_emb = self.bridge_layer[str(cascade_idx+1)](box_emb)
        else:
            raise NotImplementedError

        # return offset emb
        return boxes, offset_emb, split_lens, query_split_lens, topk_matched_idxs, box_target, box_pid, target_emb

    def get_cascade_dict(self, head_outputs):
        ###
        anchors = head_outputs['topk_anchors']
        bbox_regression = head_outputs['bbox_regression']
        matched_idxs = head_outputs['topk_matched_idxs']
        split_lens = head_outputs['split_lens']
        box_target = head_outputs['box_target']
        box_pid = head_outputs['pid_target']
        if 'target_emb' in head_outputs:
            target_emb = head_outputs['target_emb']
        else:
            target_emb = None
        if 'query_split_lens' in head_outputs:
            query_split_lens = head_outputs['query_split_lens']
        else:
            query_split_lens = None
        #
        if len(matched_idxs) == 0:
            raise Exception
        # build initial cascade dict
        cascade_dict = {
            'features': head_outputs['features'],
            'bbox_regression': bbox_regression,
            'cls_logits': head_outputs['cls_logits'],
            'anchors': anchors,
            'split_lens': split_lens,
            'query_split_lens': query_split_lens,
            'target_emb': target_emb,
            'box_target': box_target,
            'gt_emb': torch.cat(head_outputs["gt_emb"]),
            'box_pid': box_pid,
            'topk_matched_idxs': matched_idxs,
        }

        return cascade_dict

    def test_cascade(self, features, pred_emb, query_emb, pred_box, 
            image_shape):
        features_dict = dict(zip(self.featmap_names, features))
        if self.emb_align_sep:
            loc_features_dict = dict(zip(self.loc_featmap_names, features))
        else:
            loc_features_dict = features_dict
        for cascade_idx in range(self.num_cascade_steps):
            # 1) get offset embeddings
            if self.train_mode == 'qc':
                offset_emb = self.combiner.forward(query_emb, pred_emb)
            elif self.train_mode == 'oc':
                offset_emb = self.bridge_layer[str(cascade_idx+1)](pred_emb)
            else:
                raise NotImplementedError
            # 2) predict new box regs
            bbox_regression = self.regression_head[str(cascade_idx+1)](offset_emb)
            cls_logits = self.classification_head[str(cascade_idx+1)](offset_emb)
            # 3) build new boxes
            pred_box = self.box_coder.decode_single(
                bbox_regression, pred_box,
            )
            ## clip boxes to image
            pred_box = box_ops.clip_boxes_to_image(pred_box, image_shape)
            ## nms
            nms_idx = box_ops.batched_nms(pred_box, cls_logits[:, 1],
                torch.ones(pred_box.shape[0], dtype=torch.long).to(pred_box.device),
                0.5)
            pred_box = pred_box[nms_idx]
            cls_logits = cls_logits[nms_idx]
            # 4) predict new embeddings
            # Final cascade step only extracts reid embedding
            if self.emb_align_sep and (cascade_idx < (self.num_cascade_steps-1)):
                box_feat = self.loc_roi_align(loc_features_dict, [pred_box], [image_shape])
                box_emb = self.align_emb_loc_head[str(cascade_idx+2)](box_feat)
            else:
                box_feat = self.roi_align(features_dict, [pred_box], [image_shape])
                box_emb = self.align_emb_head(box_feat)
            N, D, *_ = box_emb.shape
            pred_emb = box_emb.reshape(N, D)
        return pred_box, pred_emb, cls_logits
            
    def train_cascade(self, head_outputs, targets, use_nms=False):
        # Get stuff
        cascade_dict_list = [self.get_cascade_dict(head_outputs)]
        # Iterate through n box refinement steps
        for cascade_idx in range(self.num_cascade_steps):
            curr_cascade_dict = cascade_dict_list[cascade_idx]
            # 1) get offset embeddings
            boxes, offset_emb, split_lens, query_split_lens, topk_matched_idxs, box_target, box_pid, target_emb = self.get_offset_emb(
                curr_cascade_dict, targets, cascade_idx, use_nms=use_nms)
            # 2) predict new box regs
            bbox_regression = self.regression_head[str(cascade_idx+1)](offset_emb)
            cls_logits = self.classification_head[str(cascade_idx+1)](offset_emb)

            # 4) create new head outputs
            new_cascade_dict = {
                'features': curr_cascade_dict['features'],
                'bbox_regression': bbox_regression,
                'cls_logits': cls_logits,
                'anchors': boxes,
                'split_lens': split_lens,
                'query_split_lens': query_split_lens,
                'target_emb': target_emb,
                #
                'box_pid': box_pid,
                'box_target': box_target,
                'gt_emb': curr_cascade_dict['gt_emb'],
                #
                'topk_matched_idxs': topk_matched_idxs,    
            }
            cascade_dict_list.append(new_cascade_dict)
        # Return list of cascade dicts to compute losses
        return cascade_dict_list

    def compute_loss(self, targets, head_outputs, matched_idxs, image_shapes=None):
        # type: (List[Dict[sor, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]

        ### Handle edge case with no matches
        matched_idxs = head_outputs['topk_matched_idxs']
        anchor_cls_logits = head_outputs['anchor_cls_logits']
        if len(matched_idxs) == 0:
            return {
                "anchor_classification": (anchor_cls_logits * 0.0).sum(),
            }
        cascade_dict_list = self.train_cascade(head_outputs, targets)
        ### Handle normal case
        loss_dict = {}
        if not self.reid_only:
            loss_dict["anchor_classification"] = self.anchor_classification_head.compute_loss(head_outputs)
            for cascade_idx, cascade_dict in enumerate(cascade_dict_list):
                loss_dict["bbox_regression{}".format(cascade_idx+1)] = self.regression_head[str(cascade_idx)].compute_loss(cascade_dict)
                if self.use_anchor_head or (cascade_idx > 0):
                    loss_dict["classification{}".format(cascade_idx+1)] = self.classification_head[str(cascade_idx)].compute_loss(cascade_dict)
        reid_loss = self.compute_loss_emb_tfm(head_outputs,
            cascade_dict_list[-1], targets) * self.reid_loss_weight
        loss_dict["reid"] = reid_loss
        ### Train GFN
        if self.use_gfn:
            features = dict(zip(self.featmap_names, head_outputs['features']))
            gfn_loss_dict, _ = self.gfn(features, targets,
                image_shapes)
            loss_dict.update(gfn_loss_dict)

        return loss_dict

    def filter_features(self, query_emb_per_chunk, anchor_emb_per_image, anchor_per_image, k1, image_shapes=None):
        image_shapes = torch.stack([torch.tensor(list(image_shape)) for image_shape in image_shapes]).repeat(1, 2).reshape(1, 1, 4).to(anchor_emb_per_image)
        _norm_logits_per_chunk = self.combiner.norm(query_emb_per_chunk, anchor_emb_per_image)
        ###
        anchor_logits_per_chunk, query_anchor_idx = _norm_logits_per_chunk.topk(k=k1, dim=1, largest=self.use_posnorm)
        anchor_emb_per_chunk = anchor_emb_per_image[query_anchor_idx]
        offset_emb_per_chunk = self.combiner.forward(query_emb_per_chunk.unsqueeze(1),  anchor_emb_per_chunk)
        anchor_per_chunk = anchor_per_image[query_anchor_idx]
        ###
        return offset_emb_per_chunk, anchor_per_chunk, anchor_logits_per_chunk, _norm_logits_per_chunk

    def filter_topk(self, query_emb_per_chunk, anchor_emb_per_image, anchor_per_image, k1, image_shapes=None):
        image_shapes = torch.stack([torch.tensor(list(image_shape)) for image_shape in image_shapes]).repeat(1, 2).reshape(1, 1, 4).to(anchor_emb_per_image)
        _norm_logits_per_chunk = self.combiner.norm(query_emb_per_chunk, anchor_emb_per_image)
        ###
        anchor_logits_per_chunk, query_anchor_idx = _norm_logits_per_chunk.topk(k=k1, dim=1, largest=self.use_posnorm)
        anchor_emb_per_chunk = anchor_emb_per_image[query_anchor_idx]
        offset_emb_per_chunk = self.combiner.forward(query_emb_per_chunk.unsqueeze(1),  anchor_emb_per_chunk)
        anchor_per_chunk = anchor_per_image[query_anchor_idx]
        ###
        return offset_emb_per_chunk, anchor_per_chunk, anchor_logits_per_chunk

    def filter_topk_train(self, query_emb_per_chunk, anchor_emb_per_image,
            anchor_per_image, k, mode='norm', iou=None,
            mask=None, pos_mask=None, neg_mask=None):
        #assert (iou[~mask] == 0).all()
        q = query_emb_per_chunk.shape[0]
        n, _, d = anchor_emb_per_image.shape
        if mask is None:
            mask = torch.ones(n, q, dtype=bool)
        n_idx, q_idx = torch.where(mask)
        # uses less memory
        _norm_logits_per_chunk = self.combiner.norm(query_emb_per_chunk, anchor_emb_per_image, m=mask)
        if torch.isinf(_norm_logits_per_chunk).sum() > 0:
            raise Exception
        ###
        _k = min(_norm_logits_per_chunk.shape[1], k)
        _query_anchor_idx = _norm_logits_per_chunk.topk(k=_k, dim=1, largest=self.use_posnorm).indices
        # In IoU / teacher forcing mode: replace indices for positive matches
        # (highest norm) with those with highest query-anchor IoU
        if mode == 'all':
            _query_anchor_idx = torch.arange(iou.shape[2], dtype=torch.int64)[None].repeat(mask.sum(), 1).to(iou.device)
        elif mode == 'iou':
            _k = min(iou.shape[2], k)
            _query_anchor_idx = iou.topk(k=_k, dim=2).indices[mask]
        elif mode == 'norm':
            _k = min(_norm_logits_per_chunk.shape[1], k)
            _query_anchor_idx = _norm_logits_per_chunk.topk(k=_k, dim=1, largest=self.use_posnorm).indices
        elif mode == 'iounorm':
            _iou_idx = iou.topk(k=k, dim=2).indices[mask]
            _norm_idx = _norm_logits_per_chunk.topk(k=k, dim=1, largest=self.use_posnorm).indices
            _query_anchor_idx = torch.cat([_iou_idx, _norm_idx], dim=1)
        elif mode == 'posneg':
            neg_idx = neg_mask[mask].flatten().nonzero().view(-1)
            pos_idx = pos_mask[mask].flatten().nonzero().view(-1)
            _iou_idx = iou.topk(k=k, dim=2).indices[mask]
            _pos_iou_idx = _iou_idx[pos_idx, :k//2]
            _neg_norm_idx = _norm_logits_per_chunk.topk(k=k, dim=1, largest=self.use_posnorm).indices[neg_idx]
            _pos_norm_idx = _norm_logits_per_chunk.topk(k=k, dim=1, largest=self.use_posnorm).indices[pos_idx]
            _pos_comb_idx = torch.cat([_pos_iou_idx, _pos_iou_idx, _pos_norm_idx], dim=1)
            _new_pos_norm_idx_list = []
            for __pos_comb_idx in _pos_comb_idx:
                i, c = torch.unique(__pos_comb_idx, return_counts=True)
                unused_idx = i[c==1][:k//2]
                _new_pos_norm_idx_list.append(unused_idx)
            _new_pos_norm_idx = torch.stack(_new_pos_norm_idx_list) 
            _new_pos_idx = torch.cat([_pos_iou_idx, _new_pos_norm_idx], dim=1)
            _iou_idx[pos_idx] = _new_pos_idx
            _iou_idx[neg_idx] = _neg_norm_idx
            _query_anchor_idx = _iou_idx
        ###
        anchor_emb_per_chunk = anchor_emb_per_image[n_idx].gather(1, _query_anchor_idx.unsqueeze(2).repeat(1, 1, d))
        offset_emb_per_chunk = self.combiner.forward(
            query_emb_per_chunk[q_idx].unsqueeze(1), anchor_emb_per_chunk)
        anchor_per_chunk = anchor_per_image[_query_anchor_idx]
        norm_logits_per_chunk = _norm_logits_per_chunk
        query_anchor_idx = _query_anchor_idx
        ###
        return norm_logits_per_chunk, offset_emb_per_chunk, anchor_per_chunk, query_anchor_idx

    def search_features(self, x, a=None, q=None, k=100, image_shapes=None):
        ### Compute anchor embeddings
        _anchor_emb, _ = self.feature_head(x)

        ### 243621
        query_emb = self.read_gt_embeddings(q)
        for query_emb_per_image, anchor_emb_per_image, anchor_per_image in zip(
            query_emb, _anchor_emb, a
        ):
            query_chunk_size = query_emb_per_image.shape[0]
            for query_emb_per_chunk in torch.split(query_emb_per_image, query_chunk_size):
                offset_emb_per_chunk, anchor_per_chunk, anchor_logits_per_chunk, all_offset_emb = self.filter_features(query_emb_per_chunk, anchor_emb_per_image, anchor_per_image, k, image_shapes=image_shapes)
                ###
                cls_logits = self.classification_head["0"](offset_emb_per_chunk)
                bbox_regression = self.regression_head["0"](offset_emb_per_chunk)
                anchor_cls_logits = anchor_logits_per_chunk

                # Return results
                yield k, {
                    "topk_cls_logits": cls_logits,
                    "topk_bbox_regression": bbox_regression,
                    "topk_anchors": anchor_per_chunk,
                    "topk_anchor_cls_logits": anchor_cls_logits,
                    "anchors": a,
                    "anchor_emb": _anchor_emb,
                    "offset_emb": all_offset_emb,
                }

    def search_topk(self, x, a=None, q=None, k=100, image_shapes=None):
        ### Compute anchor embeddings
        _anchor_emb, _ = self.feature_head(x)

        ### 243621
        query_emb = self.read_gt_embeddings(q)
        for query_emb_per_image, anchor_emb_per_image, anchor_per_image in zip(
            query_emb, _anchor_emb, a
        ):
            query_chunk_size = query_emb_per_image.shape[0]
            for query_emb_per_chunk in torch.split(query_emb_per_image, query_chunk_size):
                offset_emb_per_chunk, anchor_per_chunk, anchor_logits_per_chunk = self.filter_topk(query_emb_per_chunk, anchor_emb_per_image, anchor_per_image, k, image_shapes=image_shapes)
                ###
                cls_logits = self.classification_head["0"](offset_emb_per_chunk)
                bbox_regression = self.regression_head["0"](offset_emb_per_chunk)
                anchor_cls_logits = anchor_logits_per_chunk

                # Return results
                yield k, {
                    "cls_logits": cls_logits,
                    "bbox_regression": bbox_regression,
                    "anchors": anchor_per_chunk,
                    "anchor_cls_logits": anchor_cls_logits,
                }

    def search_all(self, x, q=None):
        ### Compute anchor embeddings
        _anchor_emb, shape_list = self.feature_head(x)

        ###
        query_chunk_size = 16
        query_emb = self.read_gt_embeddings(q)
        for query_emb_per_image, anchor_emb_per_image in zip(
            query_emb, _anchor_emb
        ):
            for query_emb_per_chunk in torch.split(query_emb_per_image, query_chunk_size):
                offset_emb_per_chunk = self.combiner.forward(query_emb_per_chunk.unsqueeze(1),  anchor_emb_per_image.unsqueeze(0))
                offset_emb_per_chunk = _recover_feats1(offset_emb_per_chunk, shape_list, self.num_anchors)
                # Return results
                yield {
                    "cls_logits": _recover_feats2(self.classification_head["0"](offset_emb_per_chunk), shape_list, self.num_anchors)[0],
                    "bbox_regression": _recover_feats2(self.regression_head["0"](offset_emb_per_chunk), shape_list, self.num_anchors)[0],
                    "anchor_features": _anchor_emb,
                }

    def forward(self, x, a=None, q=None, moco_features=None, 
            subsample=True, use_all_pos=False):
        ### Compute anchor embeddings
        _anchor_emb, shape_list = self.feature_head(x)

        ### Set current mode
        if self.training:
            curr_mode = self.train_mode
        else:
            curr_mode = self.test_mode

        ###
        pred_emb, query_emb = None, None
        if curr_mode == 'oc':
            if self.training:
                ## Match anchors with ground truth boxes
                matched_idxs = []
                iou_list = []
                for anchors_per_image, targets_per_image in zip(a, q):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                        )
                        iou_list.append(
                            torch.full((anchors_per_image.size(0),), 0, dtype=_anchor_emb.dtype, device=anchors_per_image.device)
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    iou_list.append(match_quality_matrix.max(dim=0).values)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))
                iou = torch.cat(iou_list)
                ## Compute offset emb
                _anchor_emb = _anchor_emb.reshape(-1, _anchor_emb.shape[-1])
                _offset_emb = self.bridge_layer["0"](_anchor_emb)
                norm_logits = torch.norm(_offset_emb, dim=1)
                ## Subsample indices used for class head
                if subsample:
                    ### Get subsample indices
                    subsample_mask, split_lens = self.fg_bg_sampler(
                        matched_idxs, scores=-norm_logits)
                    ### Subsample anchor emb
                    offset_emb = _offset_emb[subsample_mask]
                    ### Subsample anchor emb
                    _anchor_emb = _anchor_emb[subsample_mask]
                    ### Subsample matched idxs
                    matched_idxs = torch.cat(matched_idxs)
                    topk_matched_idxs = matched_idxs[subsample_mask]
                    ### Subsample ious
                    iou = iou[subsample_mask]
                    ### Subsample anchors
                    anchors = torch.cat(a)
                    topk_anchors = anchors[subsample_mask]
                ## Use only top norm indices
                else:
                    topk = 1000
                    _offset_idx = iou.topk(k=topk, dim=1, largest=True).indices
                    ## Use top norm indices and top IoU indices
                    _norm_offset_idx = norm_logits.topk(k=topk, dim=1, largest=self.use_posnorm).indices
                    _offset_idx = torch.cat([_offset_idx, _norm_offset_idx], dim=1)
                    ## Reshape offset emb
                    offset_emb = torch.gather(_offset_emb, 1, _offset_idx.unsqueeze(2).repeat(1, 1, self.emb_loc_dim))
                ## Compute class logits and bbox regression
                cls_logits = self.classification_head["0"](offset_emb)
                bbox_regression = self.regression_head["0"](offset_emb)
                #### Get query embeddings
                query_emb = self.get_gt_embeddings(x, q)
                if self.emb_align_sep:
                    query_loc_emb = self.get_gt_embeddings(x, q, emb_type='loc')
                else:
                    query_loc_emb = query_emb
                if self.use_moco:
                    moco_query_emb = self.get_gt_embeddings(moco_features, q, moco=True)
                ###
                anchor_cls_logits = self.anchor_classification_head(norm_logits)
                if not subsample:
                    matched_idxs = torch.stack(matched_idxs)
                    topk_matched_idxs = torch.gather(matched_idxs, 1, _offset_idx)
                ###
                box_target_list, pid_target_list = [], []
                for image_query, image_idx, split_len in zip(q, topk_matched_idxs.clone().split(split_lens), split_lens):
                    image_idx[image_idx<0] = 0
                    if image_query['boxes'].shape[0] == 0:
                        if subsample:
                            if use_all_pos:
                                topk_boxes = torch.zeros(0, 4).to(bbox_regression)
                            else:
                                #topk_boxes = torch.zeros(self.subsample_per_image, 4).to(bbox_regression)
                                #topk_boxes = torch.zeros(0, 4).to(bbox_regression)
                                topk_boxes = torch.zeros(split_len, 4).to(bbox_regression)
                        else:
                            topk_boxes = torch.zeros(topk*2, 4).to(bbox_regression)
                        topk_pids = torch.zeros_like(image_idx)
                    else:
                        topk_boxes = image_query['boxes'][image_idx]
                        if self.reid_objective == 'oim':
                            topk_pids = image_query['labels'][image_idx] - 1
                        else:
                            topk_pids = image_query['person_id'][image_idx]
                    box_target_list.append(topk_boxes)
                    pid_target_list.append(topk_pids)
                box_target = torch.cat(box_target_list)
                pid_target = torch.cat(pid_target_list)
                ###
                if not subsample:
                    anchors = torch.stack(a)
                    topk_anchors = torch.gather(anchors, 1, _offset_idx.unsqueeze(2).repeat(1, 1, 4))
            else:
                ##
                _offset_emb = self.bridge_layer["0"](_anchor_emb)
                norm_logits = torch.norm(_offset_emb, dim=2)
                ##
                _offset_idx = norm_logits.topk(k=1000, dim=1, largest=self.use_posnorm).indices
                offset_emb = torch.gather(_offset_emb, 1, _offset_idx.unsqueeze(2).repeat(1, 1, self.emb_loc_dim))
                ##
                cls_logits = self.classification_head["0"](offset_emb)
                bbox_regression = self.regression_head["0"](offset_emb).reshape(-1, 4)
                ##
                anchors = torch.stack(a)
                topk_anchors = torch.gather(anchors, 1, _offset_idx.unsqueeze(2).repeat(1, 1, 4)).reshape(-1, 4)
        elif curr_mode == 'qc':
            ### Compute offset emb from anchor and query embeddings
            #### Get query embeddings
            query_emb = self.get_gt_embeddings(x, q)
            if self.emb_align_sep:
                query_loc_emb = self.get_gt_embeddings(x, q, emb_type='loc')
            else:
                query_loc_emb = query_emb
            if self.use_moco:
                moco_query_emb = self.get_gt_embeddings(moco_features, q, moco=True)
            #### Compute offset
            #############################
            # clean the query dict
            good_query_emb = []
            good_query_loc_emb = []
            moco_good_query_emb = []
            for i, t in enumerate(q):
                uv, uc = t['person_id'].unique(return_counts=True)
                bad_mask = uc > 1
                if bad_mask.sum() > 0:
                    bv = uv[bad_mask]
                    good_mask = t['person_id'] != bv
                    # remove all data corresponding to the repeated person_id
                    print('NOTIFICATION: removing duplicated pid: {}'.format(bv))
                    t['person_id'] = t['person_id'][good_mask]
                    t['labels'] = t['labels'][good_mask]
                    t['boxes'] = t['boxes'][good_mask]
                else:
                    good_mask = t['person_id'] == t['person_id']
                good_query_emb.append(query_emb[i][good_mask])
                good_query_loc_emb.append(query_loc_emb[i][good_mask])
                if self.use_moco:
                    moco_good_query_emb.append(moco_query_emb[i][good_mask])
            query_emb = good_query_emb
            query_loc_emb = good_query_loc_emb
            if self.use_moco:
                moco_query_emb = moco_good_query_emb
            #
            boxes = torch.cat([t['boxes'] for t in q])
            N = len(q)
            Q = boxes.shape[0]
            K = self.subsample_per_image
            A = a[0].shape[0]
            # compute general mqm
            any_match_quality_list = []
            for _query, _anchor in zip(q, a):
                _query_boxes = _query['boxes']
                _iou = box_ops.box_iou(_query_boxes, _anchor)
                any_match_quality_list.append(_iou)
            #
            full_match_quality_matrix = torch.zeros(N, Q, A).to(boxes)
            giou_matrix = torch.zeros(N, Q, A).to(boxes)
            box_target = torch.zeros(N, Q, 4).to(boxes)
            pid_target = torch.zeros(N, Q).to(q[0]['person_id'])
            #
            iou_list = []
            labels = torch.cat([t['person_id'] for t in q])
            box_target_list = []
            for i, t in enumerate(q):
                query_idx, image_idx = torch.where(labels.unsqueeze(1) == t['person_id'].unsqueeze(0))
                iou = box_ops.box_iou(t['boxes'][image_idx], a[0])
                assert (iou.max(dim=1).values > 0).all()
                full_match_quality_matrix[i, query_idx] = iou
                #
                giou = box_ops.generalized_box_iou(t['boxes'][image_idx], a[0])
                giou_matrix[i, query_idx] = giou
                #
                box_target[i, query_idx] = t['boxes'][image_idx]
                if self.reid_objective == 'oim':
                    pid_target[i, query_idx] = (t['labels'] - 1)[image_idx]
                else:
                    pid_target[i, query_idx] = t['person_id'][image_idx]
            # get mask of query presence in each image
            matched_idxs = []
            topk_matched_idxs = []
            #
            image_query_inst_list = []
            inst_labels = torch.cat([t['person_id'] for t in q])
            for i, t in enumerate(q):
                #
                inst_mask = (inst_labels.unsqueeze(1) == t['person_id'].unsqueeze(0)).any(dim=1)
                image_query_inst_list.append(inst_mask)
                #
            image_query_inst_mask = torch.stack(image_query_inst_list)
            #
            image_query_id_mask = torch.zeros_like(image_query_inst_mask)
            j = 0
            for i, t in enumerate(q):
                l = len(t['boxes'])
                image_query_id_mask[i, j:j+l] = True
                j = j + l
            #
            assert image_query_inst_mask.shape == (N, Q)
            assert image_query_id_mask.shape == (N, Q)
            # Set the image query mask
            filter_mode = 'iou'
            pos_image_query_mask, neg_image_query_mask = None, None
            if self.match_mode == 'self':
                image_query_mask = image_query_id_mask
            elif self.match_mode == 'other':
                image_query_mask = ~image_query_id_mask & image_query_inst_mask
            elif self.match_mode == 'all':
                image_query_mask = image_query_inst_mask
            elif self.match_mode == 'algo2':
                """
                Sampling procedure (2):
                1) Sample all "other" matches
                2) Fill remaining with "self" matches
                """
                #
                other_mask = ~image_query_id_mask & image_query_inst_mask
                self_mask = image_query_id_mask
                #
                pos_image_query_mask = torch.zeros_like(image_query_id_mask)
                num_remain = self.subsample_per_query
                # 1) Sample all "other" matches
                i1, i2 = _subsample_mask(other_mask, num_remain)
                pos_image_query_mask[i1, i2] = True
                num_remain -= len(i1)
                other_num_sample = len(i1)
                # 2) Sample all "self" matches
                if num_remain > 0:
                    i1, i2 = _subsample_mask(self_mask, num_remain)
                    pos_image_query_mask[i1, i2] = True
                    num_remain -= len(i1)
                    self_num_sample = len(i1)
                else:
                    self_num_sample = 0
                # Combine image query masks
                image_query_mask = pos_image_query_mask
                # Check results
                assert (other_num_sample + self_num_sample) == image_query_mask.sum().item()
                assert (other_num_sample + self_num_sample) <= self.subsample_per_query
            elif self.match_mode == 'algo1':
                """
                Sampling procedure (1):
                1) Sample all "other" matches
                2) Sample all "self" matches
                3) Fill any remaining with negatives
                """
                #
                filter_mode = 'posneg'
                #
                other_mask = ~image_query_id_mask & image_query_inst_mask
                self_mask = image_query_id_mask
                neg_mask = ~image_query_inst_mask
                #
                pos_image_query_mask = torch.zeros_like(image_query_id_mask)
                neg_image_query_mask = torch.zeros_like(image_query_id_mask)
                num_remain = self.subsample_per_query
                # 1) Sample all "other" matches
                i1, i2 = _subsample_mask(other_mask, num_remain)
                pos_image_query_mask[i1, i2] = True
                num_remain -= len(i1)
                other_num_sample = len(i1)
                # 2) Sample all "self" matches
                if num_remain > 0:
                    i1, i2 = _subsample_mask(self_mask, num_remain)
                    pos_image_query_mask[i1, i2] = True
                    num_remain -= len(i1)
                    self_num_sample = len(i1)
                else:
                    self_num_sample = 0
                # 3) Fill any remaining with negatives
                if num_remain > 0:
                    i1, i2 = _subsample_mask(neg_mask, num_remain)
                    neg_image_query_mask[i1, i2] = True
                    num_remain -= len(i1)
                    neg_num_sample = len(i1)
                else:
                    neg_num_sample = 0
                # Combine image query masks
                image_query_mask = pos_image_query_mask | neg_image_query_mask
                # Check results
                assert (other_num_sample + self_num_sample + neg_num_sample) == image_query_mask.sum().item()
                assert (other_num_sample + self_num_sample + neg_num_sample) <= self.subsample_per_query

            # Subsample elements of the image_query_mask for memory
            if self.match_mode not in ('algo1', 'algo2'):
                iqm1, iqm2 = torch.where(image_query_mask)
                #print('Num mask {}/{}/{}:'.format(
                #    len(iqm1), self.subsample_per_query,
                #    image_query_mask.numel()))
                max_pos = self.subsample_per_query
                if len(iqm1) > max_pos:
                    num_diff = len(iqm1) - max_pos
                    rand_idx = torch.randperm(len(iqm1))[:num_diff]
                    niqm1, niqm2 = iqm1[rand_idx], iqm2[rand_idx]
                    image_query_mask[niqm1, niqm2] = False

            #
            tfm_query_loc_emb = torch.cat(query_loc_emb, dim=0)
            norm_logits, offset_emb, topk_anchors, topk_idx = self.filter_topk_train(tfm_query_loc_emb, _anchor_emb, a[0], k=K, mode=filter_mode,
                #iou=full_match_quality_matrix,
                iou=giou_matrix,
                mask=image_query_mask,
                pos_mask=pos_image_query_mask,
                neg_mask=neg_image_query_mask)
            ###
            image_query_mask_flat = image_query_mask.view(-1)
            anchor_cls_logits = self.anchor_classification_head(norm_logits)
            cls_logits = self.classification_head["0"](offset_emb)
            bbox_regression = self.regression_head["0"](offset_emb)
            # Get matched indices
            full_match_quality_matrix = full_match_quality_matrix[image_query_mask]

            #
            assert full_match_quality_matrix.shape[0] == topk_idx.shape[0], '{}, {}'.format(full_match_quality_matrix.shape, topk_idx.shape)
            ##
            if neg_image_query_mask is not None:
                neg_idx = neg_image_query_mask[image_query_mask].flatten().nonzero().view(-1)
                assert (full_match_quality_matrix[neg_idx] == 0).all()
            #
            for match_quality_matrix in full_match_quality_matrix:
                _matched_idxs = self.proposal_matcher(match_quality_matrix.unsqueeze(0))
                matched_idxs.append(_matched_idxs)
            if len(matched_idxs) > 0:
                matched_idxs = torch.stack(matched_idxs)
            else:
                matched_idxs = torch.tensor(matched_idxs)

            ## Ensure all labels for negatives are BG
            if neg_image_query_mask is not None:
                matched_idxs[neg_idx] = self.BG
                assert (matched_idxs[neg_idx] == self.BG).all()

            #any_match_quality = torch.stack(any_match_quality_list)
            n_idx = torch.where(image_query_mask)[0]
            any_match_quality = [any_match_quality_list[i] for i in n_idx.tolist()]
            ## Set all other objects in the image to nonmatch
            ## - prevents biasing object prediction to negative
            if self.match_conservative:
                new_matched_idxs = []
                for _matched_idxs, _any_iou in zip(
                        matched_idxs, any_match_quality):
                    if _any_iou.shape[0] > 0:
                        _any_matched_idxs = self.proposal_matcher(_any_iou)
                        any_mask = _any_matched_idxs >= self.FG
                        this_mask = _matched_idxs >= self.FG
                        final_mask = any_mask & ~this_mask
                        _matched_idxs[final_mask] = self.NM
                        new_matched_idxs.append(_matched_idxs)
                    else:
                        _matched_idxs[:] = self.BG
                        new_matched_idxs.append(_matched_idxs)
                if len(new_matched_idxs) > 0:
                    matched_idxs = torch.stack(new_matched_idxs)
                else:
                    matched_idxs = torch.tensor(new_matched_idxs)

            ## Get top-k from each set of labels
            for _matched_idxs, _topk_idx in zip(matched_idxs, topk_idx):
                _topk_matched_idxs = _matched_idxs[_topk_idx]
                
                topk_matched_idxs.append(_topk_matched_idxs)
            if len(topk_matched_idxs) > 0:
                topk_matched_idxs = torch.stack(topk_matched_idxs)
            else:
                topk_matched_idxs = torch.tensor(topk_matched_idxs)
            #print('bg, nm, fg:', (topk_matched_idxs==self.BG).sum().item(), (topk_matched_idxs==self.NM).sum().item(), (topk_matched_idxs>=self.FG).sum().item())

            # Set idx to background when query is not in image
            if self.match_mode in ('self', 'other', 'algo2'):
                # Set all non-instance matches to background
                matched_idxs[~image_query_inst_mask.view(-1)[image_query_mask_flat]] = self.BG
                topk_matched_idxs[~image_query_inst_mask.view(-1)[image_query_mask_flat]] = self.BG
            if self.match_mode == 'self':
                # Set all "other" matches to non-match
                other_mask = ~image_query_id_mask & image_query_inst_mask
                matched_idxs[other_mask.view(-1)[image_query_mask_flat]] = self.NM
                topk_matched_idxs[other_mask.view(-1)[image_query_mask_flat]] = self.NM
            elif self.match_mode == 'other':
                # Set all id matches to non-match
                matched_idxs[image_query_id_mask.view(-1)[image_query_mask_flat]] = self.NM
                topk_matched_idxs[image_query_id_mask.view(-1)[image_query_mask_flat]] = self.NM
            elif self.match_mode == 'all':
                pass
            
            ### Flatten and get split lens
            split_lens = (image_query_mask.sum(dim=1) * K).tolist()
            topk_matched_idxs = topk_matched_idxs.view(-1)
            cls_logits = cls_logits.view(-1, 2)
            anchor_cls_logits = anchor_cls_logits.reshape(-1, 2)
            bbox_regression = bbox_regression.view(-1, 4)
            matched_idxs = matched_idxs.view(-1)
            topk_anchors = topk_anchors.view(-1, 4)
            full_match_quality_matrix = full_match_quality_matrix.view(-1)
            topk_idx = topk_idx.view(-1)

            ### Query split lens
            query_split_lens = image_query_mask.sum().item()*[K]

            ### Prep target emb, box
            target_emb = tfm_query_loc_emb.unsqueeze(0).repeat(N, 1, 1)[image_query_mask].unsqueeze(1).repeat(1, K, 1).reshape(-1, tfm_query_loc_emb.shape[-1])
            box_target = box_target[image_query_mask].unsqueeze(1).repeat(1, K, 1).reshape(-1, 4)
            pid_target = pid_target[image_query_mask].unsqueeze(1).repeat(1, K).reshape(-1)

        ### Return dict of results
        if self.training:
            if curr_mode == 'oc':
                output_dict = {
                    "features": x,
                    "cls_logits": cls_logits,
                    "anchor_cls_logits": anchor_cls_logits,
                    "bbox_regression": bbox_regression,
                    "box_target": box_target,
                    "pid_target": pid_target,
                    "gt_emb": query_emb,
                    "matched_idxs": matched_idxs,
                    "topk_matched_idxs": topk_matched_idxs,
                    "topk_anchors": topk_anchors,
                    "split_lens": split_lens,
                    ###
                    "gt_loc_emb": query_loc_emb,
                    "anchors": torch.stack(a),
                    ### unique to oc mode
                    "embeddings": pred_emb,
                    "offset_emb": offset_emb.reshape(-1, 128),
                }
            elif curr_mode == 'qc':
                output_dict = {
                    "features": x,
                    "cls_logits": cls_logits,
                    "anchor_cls_logits": anchor_cls_logits,
                    "bbox_regression": bbox_regression,
                    "box_target": box_target,
                    "pid_target": pid_target,
                    "gt_emb": query_emb,
                    "matched_idxs": matched_idxs,
                    "topk_matched_idxs": topk_matched_idxs,
                    "topk_anchors": topk_anchors,
                    "split_lens": split_lens,
                    ###
                    "gt_loc_emb": query_loc_emb,
                    "iou": full_match_quality_matrix,
                    "anchors": torch.stack(a),
                    "topk_idx": topk_idx,
                    ### unique to qc mode
                    "mask": image_query_mask,
                    ### NEW
                    "target_emb": target_emb,
                    "query_split_lens": query_split_lens,
                }
            if self.use_moco:
                output_dict.update({
                    'moco_features': moco_features,
                    'moco_gt_emb': moco_query_emb,
                })
            return output_dict
        else:
            if curr_mode == 'oc':
                output_dict = {
                    "cls_logits": cls_logits,
                    "bbox_regression": bbox_regression,
                    "topk_anchors": topk_anchors,
                }
            else:
                output_dict = {
                    "cls_logits": _recover_feats2(self.classification_head(offset_emb), shape_list, self.num_anchors)[0],
                    "bbox_regression": _recover_feats2(self.regression_head(offset_emb), shape_list, self.num_anchors)[0], 
                    "embeddings": pred_emb,
                    "anchor_features": _anchor_emb,
                }
                output_dict["anchor_cls_logits"] = _recover_feats2(self.anchor_classification_head(offset_emb), shape_list, self.num_anchors)[0]
            return output_dict

class NoScaleLogits(nn.Module):
    def __init__(self, posnorm=False):
        super().__init__()
        self.posnorm = posnorm

    def forward(self, norm, c=1e6):
        # Stack logits with default large negative value for 0
        if self.posnorm:
            sig = torch.stack([-torch.ones_like(norm)*c, norm], dim=-1)
        else:
            sig = torch.stack([-torch.ones_like(norm)*c, -norm], dim=-1)
        # Return stacked logits
        return sig

class FixedNormLogits(nn.Module):
    """
    Parameterization is different from the paper but the result is the same:
    z = -(w - u)/s
    """
    def __init__(self, dim, corrected=True, posnorm=False):
        super().__init__()
        self.corrected = corrected
        self.posnorm = posnorm
        # Corrected NAE
        if self.corrected:
            self.alpha = self._compute_alpha(dim)
            self.beta = self._compute_beta(dim, self.alpha)
        else:
            raise Exception

    def _compute_alpha(self, d):
        sqrt_a = math.sqrt(2) * scipy.special.gamma((d+1)/2) / scipy.special.gamma(d/2)
        a = sqrt_a ** 2
        # Fallback approximation, true for large d
        if math.isnan(a) and (d > 256):
            a = d - 0.5
        return a

    def _compute_beta(self, d, a):
        b = math.sqrt(a / (d - a))
        return b

    def forward(self, norm, c=1e6):
        # Correct the norm if needed
        if self.corrected:
            _norm = (norm / math.sqrt(self.alpha)).float()
            _sig = self.beta * (_norm - 1)
        else:
            _sig = norm
        # Stack logits with default large negative value for 0
        if self.posnorm:
            sig = torch.stack([-torch.ones_like(_sig)*c, _sig], dim=-1)
        else:
            sig = torch.stack([-torch.ones_like(_sig)*c, -_sig], dim=-1)
        # Return stacked logits
        return sig

class BatchNormLogits(nn.Module):
    def __init__(self, posnorm=False):
        super().__init__()
        self.norm = SafeBatchNorm1d(1)
        self.posnorm = posnorm

    def forward(self, x, c=1e6):
        s = x.shape
        x = x.reshape(-1, 1)
        x = self.norm(x)
        x = x.reshape(s)
        # Stack logits with default large negative value for 0
        if self.posnorm:
            sig = torch.stack([-torch.ones_like(x)*c, x], dim=-1)
        else:
            sig = torch.stack([-torch.ones_like(x)*c, -x], dim=-1)
        # Return stacked logits
        return sig

class L2Norm(nn.Module):
    def forward(self, x):
        return x.norm(dim=-1)

class SPNetClassificationHead(nn.Module):
    """
    A classification head for use in SPNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        focal_alpha=0.5,
        focal_gamma=2.0,
        loss_func='focal',
        logits='mlp',
        pos_smoothing=None,
        use_posnorm=False,
    ):
        super().__init__()
        self.pos_smoothing = pos_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_func = loss_func
        if logits == 'mlp':
            layers = []
            for _ in range(4):
                layers.append(nn.Linear(in_channels, in_channels))
                layers.append(nn.ReLU())
            layers.append(L2Norm())
            self.tfm = nn.Sequential(*layers)
            self.head = FixedNormLogits(in_channels, posnorm=use_posnorm)
        elif logits == 'noscale':
            self.tfm = L2Norm()
            self.head = NoScaleLogits(posnorm=use_posnorm)
        elif logits == 'norm':
            self.tfm = L2Norm()
            self.head = FixedNormLogits(in_channels, posnorm=use_posnorm)
        elif logits == 'batchnorm':
            self.tfm = L2Norm()
            self.head = BatchNormLogits(posnorm=use_posnorm)
        else:
            raise NotImplementedError

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, cascade_dict):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        matched_idxs = cascade_dict['topk_matched_idxs'].reshape(-1)
        valid_mask = matched_idxs != -2
        #
        anchor_cls_logits = cascade_dict['cls_logits']
        if len(matched_idxs) == 0:
            return torch.tensor(0.0).to(anchor_cls_logits)
        anchor_cls_labels = matched_idxs.clone().unsqueeze(1)
        anchor_cls_labels[matched_idxs>=0] = 1
        anchor_cls_labels[matched_idxs==-1] = 0
        anchor_cls_labels = torch.cat([torch.zeros_like(anchor_cls_labels), anchor_cls_labels], dim=1).float()

        # filter valid samples
        anchor_cls_logits = anchor_cls_logits[valid_mask]
        anchor_cls_labels = anchor_cls_labels[valid_mask]

        # use only hardest negatives
        with torch.no_grad():
            pos_mask = anchor_cls_labels[:, 1] == 1
            num_pos = pos_mask.sum().item()

        # positive smoothing
        if self.pos_smoothing is not None:
            anchor_cls_labels[anchor_cls_labels==1] -= self.pos_smoothing

        # compute the classification loss
        if self.loss_func == 'xe':
            _loss = F.binary_cross_entropy_with_logits(
                anchor_cls_logits,
                anchor_cls_labels,
                reduction="none",
            ) / max(1, num_pos)
        elif self.loss_func == 'focal':
            _loss = sigmoid_focal_loss(
                anchor_cls_logits,
                anchor_cls_labels,
                reduction="none",
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ) / max(1, num_pos)
        assert not torch.isnan(_loss).any()
        loss = _loss.sum()
        return loss

    def forward(self, x):
        x = self.tfm(x)
        x = self.head(x)
        return x

class SPNetAnchorClassificationHead(nn.Module):
    """
    A classification head for use in SPNet.

    Args:
        in_features (int): number of channels of the input feature
    """

    def __init__(
        self,
        in_features,
        focal_alpha=0.25,
        focal_gamma=2.0,
        temp=0.5,
        logits='norm',
        pos_smoothing=None,
        use_posnorm=False,
    ):
        super().__init__()
        self.pos_smoothing = pos_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if logits == 'noscale':
            self.logits = NoScaleLogits(posnorm=use_posnorm)
        elif logits == 'norm':
            self.logits = FixedNormLogits(in_features, posnorm=use_posnorm)
        elif logits == 'batchnorm':
            self.logits = BatchNormLogits(posnorm=use_posnorm)
        else:
            raise Exception

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, head_outputs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        anchor_cls_logits = head_outputs['anchor_cls_logits']
        matched_idxs = head_outputs['matched_idxs']
        if len(matched_idxs) == 0:
            return (anchor_cls_logits * 0.0).sum()
        anchor_cls_labels = matched_idxs.clone().unsqueeze(1)
        valid_mask = matched_idxs != -2
        anchor_cls_labels[matched_idxs>=0] = 1
        anchor_cls_labels[matched_idxs==-1] = 0
        anchor_cls_labels = torch.cat([torch.zeros_like(anchor_cls_labels), anchor_cls_labels], dim=1).float()

        # filter valid samples
        anchor_cls_logits = anchor_cls_logits[valid_mask]
        anchor_cls_labels = anchor_cls_labels[valid_mask]

        # use only hardest negatives
        with torch.no_grad():
            pos_mask = anchor_cls_labels[:, 1] == 1
            num_pos = pos_mask.sum().item()

        # positive smoothing
        if self.pos_smoothing is not None:
            anchor_cls_labels[anchor_cls_labels==1] -= self.pos_smoothing

        # compute the classification loss
        _loss = sigmoid_focal_loss(
            anchor_cls_logits[:, 1],
            anchor_cls_labels[:, 1],
            reduction="none",
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ) / max(1, num_pos)
        assert not torch.isnan(_loss).any()
        loss = _loss.sum()
        return loss

    def forward(self, x):
        x = self.logits(x)
        return x

class SPNetRegressionHead(nn.Module):
    """
    A regression head for use in SPNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(self, in_channels):
        super().__init__()

        layers = []
        for _ in range(4):
            layers.append(nn.Linear(in_channels, in_channels))
            layers.append(nn.ReLU())
        self.tfm = nn.Sequential(*layers)

        self.head = nn.Linear(in_channels, 4)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self._loss_type = "giou"

    def compute_loss(self, cascade_dict):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        matched_idxs = cascade_dict['topk_matched_idxs']
        match_mask = (matched_idxs>=0).reshape(-1)
        anchors = cascade_dict['anchors'][match_mask]
        bbox_regression = cascade_dict['bbox_regression'][match_mask]
        box_target = cascade_dict['box_target'][match_mask]

        # compute the loss
        loss = _box_loss(
                self._loss_type,
                self.box_coder,
                anchors,
                box_target,
                bbox_regression,
                reduction='mean',
            )

        return loss

    def forward(self, x):
        x = self.tfm(x)
        x = self.head(x)
        return x

class SPNetFeatureHead(nn.Module):
    """
    A classification head for use in SPNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_features (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_features,
        prior_probability=0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        self.emb_pred = nn.Conv2d(in_channels, num_anchors * num_features, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.emb_pred.weight, std=0.01)
        torch.nn.init.constant_(self.emb_pred.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_features = num_features
        self.num_anchors = num_anchors

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_emb = []

        shape_list = []
        for features in x:
            emb = self.emb_pred(features)

            # Permute classification output from (N, AD, H, W) to (N, HWA, D).
            N, _, H, W = emb.shape
            emb = emb.view(N, -1, self.num_features, H, W)
            emb = emb.permute(0, 3, 4, 1, 2)
            emb = emb.reshape(N, -1, self.num_features)  # Size=(N, HWA, D)

            all_emb.append(emb)
            shape_list.append((H, W))

        return torch.cat(all_emb, dim=1), shape_list


class SPNet(nn.Module):
    """
    Implements SPNet.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        config,
        backbone,
        num_classes,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.5,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=300,
        **kwargs,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.use_classifier_train = config['use_classifier_train']
        self.use_classifier_test = config['use_classifier_test']
        self.classifier_objective = config['classifier_objective']
        self.emb_align_mode = config['emb_align_mode']
        self.emb_align_sep = config['emb_align_sep']
        self.store_anchors = config['compute_anchor_metrics']
        self.num_cascade_steps = config['num_cascade_steps']

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone
        self.freeze_backbone = config['freeze_backbone']
    
        # MOCO
        self.use_moco = config['use_moco']
        self.moco_copy_teacher = config['moco_copy_teacher']
        if self.use_moco or self.moco_copy_teacher:
            self.moco_backbone = copy.deepcopy(backbone) 
            self.moco_backbone.requires_grad_(False)
            self.moco_backbone.use_classifier = False

        # Knowledge Preservation (lwf) backbone
        if self.classifier_objective == 'lwf':
            print('==> setting up lwf backbone...')
            self.lwf_backbone = copy.deepcopy(backbone)
            ## No updates to this backbone
            for p in self.lwf_backbone.parameters():
                p.requires_grad = False

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None instead of {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.transform = GeneralizedRCNNTransform()

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    def compute_loss(self, targets, head_outputs, anchors, image_shapes=None):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, matched_idxs, image_shapes=image_shapes)

    def postprocess_detections_emb(self, head_outputs, anchors, image_shapes, features, index_by_query=False):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)
        assert num_images == 1
        image_index = 0
        num_queries = len(class_logits[0])

        detections: List[Dict[str, Tensor]] = []

        m, t, p, b = 0, 0, 0, 0
        for query_index in range(num_queries):
            box_regression_per_image = [br[query_index] for br in box_regression]
            logits_per_image = [cl[query_index] for cl in class_logits]
            image_shape = image_shapes[image_index]
            if index_by_query:
                anchors_per_image = [anchor[query_index] for anchor in anchors]
            else:
                anchors_per_image = anchors[image_index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                t += scores_per_level.shape[0]
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                m += len(idxs)

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            p += len(image_boxes)
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            b += len(keep)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        # Extract embeddings
        assert len(image_shapes) == 1
        boxes = torch.cat([d['boxes'] for d in detections])
        box_lens = [len(d['boxes']) for d in detections]
        boxes_dict = [{
            'boxes': boxes,
            'image_shape': image_shapes[0],
        }]
        emb = self.head.get_gt_embeddings(features, boxes_dict)
        emb_list = torch.split(emb[0], box_lens)
        for e, d in zip(emb_list, detections):
            d['embeddings'] = e

        # Return list of detections
        return detections

    def postprocess_detections_tfm_oc(self, head_outputs, image_shapes, features):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        anchors = head_outputs["topk_anchors"]

        num_images = len(image_shapes)
        assert num_images == 1
        image_shape = image_shapes[0]

        detections: List[Dict[str, Tensor]] = []

        box_regression_per_level = box_regression
        logits_per_level = class_logits
        anchors_per_level = anchors
        #
        num_classes = logits_per_level.shape[-1]

        # remove low scoring boxes
        scores_per_level = torch.sigmoid(logits_per_level).flatten()
        keep_idxs = scores_per_level > self.score_thresh
        scores_per_level = scores_per_level[keep_idxs]
        topk_idxs = torch.where(keep_idxs)[0]

        # keep only topk scoring predictions
        num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
        scores_per_level, idxs = scores_per_level.topk(num_topk)
        topk_idxs = topk_idxs[idxs]

        anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
        labels_per_level = topk_idxs % num_classes

        boxes_per_level = self.box_coder.decode_single(
            box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
        )
        boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

        image_boxes = boxes_per_level
        image_scores = scores_per_level
        image_labels = labels_per_level

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "cws": image_scores[keep],
                "labels": image_labels[keep],
            }
        )
        # Extract embeddings
        assert len(image_shapes) == 1
        boxes = torch.cat([d['boxes'] for d in detections])
        box_lens = [len(d['boxes']) for d in detections]
        boxes_dict = [{
            'boxes': boxes,
            'image_shape': image_shapes[0],
        }]
        if self.num_cascade_steps > 0:
            emb = self.head.get_gt_embeddings(features, boxes_dict, emb_type='loc')
        else:
            emb = self.head.get_gt_embeddings(features, boxes_dict)
        emb_list = torch.split(emb[0], box_lens)
        for e, d in zip(emb_list, detections):
            d['embeddings'] = e

        # Optional cascaded box refinement
        if self.num_cascade_steps > 0:
            for d in detections:
                if d['scores'].shape[0] > 0:
                    new_box, new_emb, cls_logits = self.head.test_cascade(features,
                        d['embeddings'], None, d['boxes'], image_shapes[0])
                    d['boxes'] = new_box
                    d['embeddings'] = new_emb
                    d['scores'] = torch.sigmoid(cls_logits.max(dim=1).values)
                    # Use final stage scores for CWS by default
                    d['cws'] = d['scores']
                    d['labels'] = cls_logits.max(dim=1).indices

        return detections

    def postprocess_detections_emb_query(self, head_outputs, anchors, image_shapes,
            features,
            use_sim_as_det_score=False, queries=None, top_only=False, store_anchors=True):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        num_images = len(image_shapes)
        assert num_images == 1
        anchor_class_logits = head_outputs["anchor_cls_logits"][0]
        class_logits = head_outputs["cls_logits"][0]
        box_regression = head_outputs["bbox_regression"][0]

        image_index = 0
        image_shape = image_shapes[image_index]

        detections: List[Dict[str, Tensor]] = []

        box_regression_per_level = box_regression
        logits_per_level = class_logits
        anchor_logits_per_level = anchor_class_logits
        orig_anchors_per_level = anchors[0].clone()
        anchors_per_level = anchors[0]
        Q, A, _ = anchors_per_level.shape

        num_classes = logits_per_level.shape[-1]

        # keep only topk scoring predictions
        if self.topk_candidates < A:
            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).reshape(Q, -1)
            keep_mask = scores_per_level > self.score_thresh
            scores_per_level[~keep_mask] = 0.0

            #
            scores_per_level, topk_idxs = scores_per_level.topk(self.topk_candidates, dim=1)

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes
            _anchor_idxs = anchor_idxs.unsqueeze(2).repeat(1, 1, 4)
            box_regression_per_level = box_regression_per_level.gather(1, _anchor_idxs)
            anchors_per_level = anchors_per_level.gather(1, _anchor_idxs)

            # not implemented?
            anchor_scores_per_level = torch.sigmoid(-anchor_logits_per_level).reshape(Q, -1)
            anchor_scores_per_level = anchor_scores_per_level.gather(1, anchor_idxs)
        else:
            scores_per_level, labels_per_level = torch.sigmoid(logits_per_level).max(dim=2)
            anchor_scores_per_level = torch.sigmoid(anchor_logits_per_level)

        image_labels = labels_per_level
        scores_per_level = scores_per_level.reshape(-1)
        anchor_scores_per_level = anchor_scores_per_level.reshape(-1)
        score_mask = scores_per_level > self.score_thresh
        image_scores = scores_per_level[score_mask]
        image_anchor_scores = anchor_scores_per_level[score_mask]
        K = box_regression_per_level.shape[1]
        box_regression_per_level = box_regression_per_level.reshape(-1, 4)[score_mask]
        anchors_per_level = anchors_per_level.reshape(-1, 4)[score_mask]

        ###
        label_offsets = (torch.arange(Q).unsqueeze(1).repeat(1, K).reshape(-1).to(image_labels) * num_classes)[score_mask]
        image_index = torch.arange(Q).unsqueeze(1).repeat(1, K).reshape(-1).to(image_labels)[score_mask]
        image_labels = image_labels.reshape(-1)[score_mask] + label_offsets
        
        assert box_regression_per_level.shape == anchors_per_level.shape
        image_boxes = self.box_coder.decode_single(
            box_regression_per_level, anchors_per_level
        )
        image_boxes = box_ops.clip_boxes_to_image(image_boxes, image_shape)
        image_anchor_boxes = box_ops.clip_boxes_to_image(anchors_per_level, image_shape)
        ###

        # non-maximum suppression
        assert image_boxes.shape[0] == image_scores.shape[0] == image_labels.shape[0], f'{image_boxes.shape[0]=} == {image_scores.shape[0]=} == {image_labels.shape[0]=}'
        _keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = _keep.sort().values

        keep_image_index = image_index[keep]
        split_sections_dict = dict(zip(range(Q), Q*[0]))
        unique_vals, unique_counts = keep_image_index.unique(return_counts=True)
        _split_sections_dict = dict(zip(unique_vals.tolist(), unique_counts.tolist()))
        split_sections_dict = {**split_sections_dict, **_split_sections_dict}
        split_sections = list(split_sections_dict.values())
        
        assert sum(split_sections) == len(keep)
        boxes = image_boxes[keep].split(split_sections)
        anchor_boxes = image_anchor_boxes[keep].split(split_sections)
        scores = image_scores[keep].split(split_sections)
        anchor_scores = image_anchor_scores[keep].split(split_sections)
        labels = (image_labels - label_offsets)[keep].split(split_sections)

        if store_anchors:
            for _boxes, _scores, _labels, _anchors, _anchor_scores, _anchor_boxes in zip(boxes, scores, labels, orig_anchors_per_level, anchor_scores, anchor_boxes):
                detections.append(
                    {
                        "boxes": _boxes,
                        "scores": _scores,
                        "labels": _labels,
                        "anchors": _anchors,
                        "anchor_scores": _anchor_scores,
                        "anchor_boxes": _anchor_boxes,
                    }
                )
        else:
            for _boxes, _scores, _labels in zip(boxes, scores, labels):
                detections.append(
                    {
                        "boxes": _boxes,
                        "scores": _scores,
                        "cws": _scores,
                        "labels": _labels,
                    }
                )

        # Extract embeddings
        assert len(image_shapes) == 1
        boxes = torch.cat([d['boxes'] for d in detections])
        box_lens = [len(d['boxes']) for d in detections]
        boxes_dict = [{
            'boxes': boxes,
            'image_shape': image_shapes[0],
        }]

        ## reid embeddings
        if self.num_cascade_steps > 0:
            emb = self.head.get_gt_embeddings(features, boxes_dict, emb_type='loc')
        else:
            emb = self.head.get_gt_embeddings(features, boxes_dict)
        emb_list = torch.split(emb[0], box_lens)
        for e, d in zip(emb_list, detections):
            d['embeddings'] = e

        ###
        if use_sim_as_det_score:
            query_emb = queries[0]['query_emb']
            for q, e, d in zip(query_emb, emb_list, detections):
                d['scores'] = torch.sigmoid((F.normalize(e, dim=1)@F.normalize(q, dim=1).T).squeeze(1)) #* d['scores']

        if top_only:
            for d in detections:
                if d['scores'].shape[0] > 0:
                    top_score_idx = d['scores'].argmax()
                    d['scores'] = d['scores'][top_score_idx].unsqueeze(0)
                    d['boxes'] = d['boxes'][top_score_idx].unsqueeze(0)
                    d['labels'] = d['labels'][top_score_idx].unsqueeze(0)
                    d['embeddings'] = d['embeddings'][top_score_idx].unsqueeze(0)

        # Optional cascaded box refinement
        if self.num_cascade_steps > 0:
            query_loc_emb = queries[0]['query_loc_emb']
            for q, d in zip(query_loc_emb, detections):
                if d['scores'].shape[0] > 0:
                    new_box, new_emb, cls_logits = self.head.test_cascade(
                        features, d['embeddings'], q,    d['boxes'], image_shapes[0])
                    d['boxes'] = new_box
                    d['embeddings'] = new_emb
                    d['scores'] = torch.sigmoid(cls_logits.max(dim=1).values)
                    # Use final stage scores for CWS by default
                    d['cws'] = d['scores']
                    d['labels'] = cls_logits.max(dim=1).indices
                else:
                    d['embeddings'] = torch.zeros(0, queries[0]['query_emb'][0].shape[-1]).to(query_loc_emb[0])

        # Return list of detections
        return detections

    def forward(self, images, targets=None, queries=None, inference_mode='both'):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        image_shapes = images.image_sizes
        
        # Check for degenerate boxes
        if targets is not None:
            # store image shapes in targets
            for t, s in zip(targets, image_shapes):
                t['image_shape'] = s
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # show the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone (and optional class_logits)
        if self.use_classifier_train:
            if self.use_classifier_test and (not self.training):
                class_logits = self.backbone(images.tensors, shortcut=True)
                return class_logits
            else:
                features, class_logits = self.backbone(images.tensors)
        else:
            features = self.backbone(images.tensors)

        # restructure features if needed
        if 'pool' in features:
            features.pop('pool')
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # List of features
        feature_keys = features.keys()
        features = list(features.values())

        # Get MOCO features
        if self.use_moco:
            moco_features = self.moco_backbone(images.tensors)

            # restructure features if needed
            if 'pool' in features:
                moco_features.pop('pool')
            if isinstance(features, torch.Tensor):
                moco_features = OrderedDict([("0", moco_features)])

            # List of moco features
            moco_features = list(moco_features.values())
        else:
            moco_features = None

        losses = {}
        #detections: List[Dict[str, Tensor]] = []
        output_list = [{} for i in range(len(images.tensors))]
        if self.training:
            # create the set of anchors
            anchors = self.anchor_generator(images, features)

            # compute classifier loss
            if self.use_classifier_train:
                if self.classifier_objective == 'lwf':
                    with torch.no_grad():
                        _, lwf_logits = self.lwf_backbone(
                            images.tensors, shortcut=False)
                else:
                    lwf_logits = None
                classifier_loss = self.head.compute_loss_classifier(
                    class_logits, targets, lwf_logits=lwf_logits)
                losses[f'scene_classification_{self.classifier_objective}'] = classifier_loss

            # compute the spnet heads outputs using the features
            head_outputs = self.head(features, a=anchors, q=targets, moco_features=moco_features)

            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses.update(self.compute_loss(targets, head_outputs, anchors, image_shapes=image_shapes))
            return losses, head_outputs

        if inference_mode in ('gt', 'both'):
            assert targets is not None
            gt_emb_list = self.head.get_gt_embeddings(features, targets)
            for i, gt_emb in enumerate(gt_emb_list):
                output_list[i]['gt_emb'] = gt_emb
            if self.emb_align_sep:
                gt_loc_emb_list = self.head.get_gt_embeddings(features, targets, emb_type='loc')
                for i, gt_loc_emb in enumerate(gt_loc_emb_list):
                    output_list[i]['gt_loc_emb'] = gt_loc_emb
            else:
                for i, gt_emb in enumerate(gt_emb_list):
                    output_list[i]['gt_loc_emb'] = gt_emb

        if inference_mode == 'search_features':
            # create the set of anchors
            anchors = self.anchor_generator(images, features)

            # compute the spnet heads outputs using the features
            output_list = []
            for num_anchors_per_level, head_outputs in self.head.search_features(features, a=anchors, q=queries, image_shapes=image_shapes):
                output_list.append(head_outputs)
            return output_list
            
        if inference_mode == 'search_topk':
            # create the set of anchors
            anchors = self.anchor_generator(images, features)

            # compute the spnet heads outputs using the features
            output_list = []
            for num_anchors_per_level, head_outputs in self.head.search_topk(features, a=anchors, q=queries, image_shapes=image_shapes):
                # split outputs per level
                split_head_outputs: Dict[str, List[Tensor]] = {}
                for k in head_outputs:
                    if k not in ('features', 'gt_emb', 'gt_loc_emb'):
                        if (head_outputs[k] is not None) and (type(head_outputs[k]) != tuple):
                            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))

                # compute the detections
                detections = self.postprocess_detections_emb_query(split_head_outputs, split_head_outputs['anchors'], image_shapes, features, queries=queries, store_anchors=self.store_anchors)
                for i, d in enumerate(detections):
                    new_d = {
                        'det_boxes': d['boxes'],
                        'det_scores': d['scores'],
                        'det_cws': d['cws'],
                        'det_labels': d['labels'],
                        'det_emb': d['embeddings'],
                        #'det_loc_emb': d['loc_embeddings'],
                    }
                    if self.store_anchors:
                        new_d['det_anchors'] = d['anchors']
                        new_d['det_anchor_boxes'] = d['anchor_boxes']
                        new_d['det_anchor_scores'] = d['anchor_scores']
                    output_list.append(new_d)
            output_list = [output_list]

        if inference_mode == 'search_all':
            # create the set of anchors
            anchors = self.anchor_generator(images, features)

            # compute the spnet heads outputs using the features
            output_list = []
            for head_outputs in self.head.search_all(features, q=queries):
                # recover level sizes
                num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
                HW = 0
                for v in num_anchors_per_level:
                    HW += v
                HWA = head_outputs["cls_logits"].size(1)
                A = HWA // HW
                num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

                # split outputs per level
                split_head_outputs: Dict[str, List[Tensor]] = {}
                for k in head_outputs:
                    if k not in ('features', 'gt_emb', 'anchor_features'):
                        if (head_outputs[k] is not None) and (type(head_outputs[k]) != tuple):
                            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
                split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

                # compute the detections
                detections = self.postprocess_detections_emb(split_head_outputs, split_anchors, image_shapes, features)
                for i, d in enumerate(detections):
                    output_list.append({
                        'det_boxes': d['boxes'],
                        'det_scores': d['scores'],
                        'det_labels': d['labels'],
                        'det_emb': d['embeddings'],#*d['scores'].unsqueeze(1),
                    })
            output_list = [output_list]

        if inference_mode in ('det', 'both'):
            # create the set of anchors
            anchors = self.anchor_generator(images, features)
            # compute the spnet heads outputs using the features
            head_outputs = self.head(features, a=anchors)

            # recover level sizes
            detections = self.postprocess_detections_tfm_oc(
                head_outputs, image_shapes, features)

            # Get image features from the GFN
            if self.head.use_gfn:
                scene_emb = self.head.gfn.get_scene_emb(
                    dict(zip(feature_keys, features))).split(1, 0)
            else:
                scene_emb = [torch.empty(0) for _ in range(len(image_shapes))]

            for i, d in enumerate(detections):
                output_list[i] = {**{
                    'det_boxes': d['boxes'],
                    'det_scores': d['scores'],
                    'det_cws': d['cws'],
                    'det_labels': d['labels'],
                    'det_emb': d['embeddings'],#*d['scores'].unsqueeze(1),
                    'scene_emb': scene_emb[i],
                }, **output_list[i]}

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SPNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return output_list


def spnet(
    config,
    oim_lut_size: int = None,
    progress: bool = True,
    num_classes: Optional[int] = 2,
    **kwargs: Any,
) -> SPNet:
    """
    Constructs an SPNet model with a FPN backbone.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
    """

    # Backbone weights
    if config['backbone_weights'] == 'random':
        weights_backbone = None
    elif config['backbone_weights'] == 'in1k':
        weights_backbone = 'IMAGENET1K_V1'

    # Setup backbone and neck
    emb_head = None
    if config['neck_type'] == 'fpn':
        if config['model'] == 'vit':
            if config['backbone_arch'] == 'vit-s16':
                if config['backbone_weights'] == 'in1k':
                    trunk = timm.create_model(
                        'vit_small_patch16_224.augreg_in1k',
                        img_size=config['aug_crop_res'],
                        checkpoint_path='/remote_logging/checkpoints/timm/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
                else:
                    trunk = timm.create_model(
                        'vit_small_patch16_224',
                        img_size=config['aug_crop_res'],
                        pretrained=False)
                fpn_dim = 384
            else:
                raise Exception
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(VitHead(trunk, fpn_dim))
            else:
                fpn_dim = 384
            # Create backbone
            backbone = VitSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim)
        elif config['model'] == 'pass_vit':
            if config['backbone_arch'] == 'vit-s16':
                trunk = PASSViT_small(img_size=(config['aug_crop_res'],config['aug_crop_res']))
                if config['backbone_weights'] == 'in1k':
                    #ckpt_path = '/remote_logging/checkpoints/timm/vit_small_p16_224-15ec54c9.pth'
                    ckpt_path = '/remote_logging/checkpoints/timm/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
                    #trunk.load_param(ckpt_path)
                    trunk._load_weights(ckpt_path)
                elif config['backbone_weights'] == 'pass':
                    ckpt_path = '/remote_logging/checkpoints/pass/pass_vit_small.pth'
                    trunk.load_param(ckpt_path)
                fpn_dim = 384
            elif config['backbone_arch'] == 'vit-b16':
                trunk = PASSViT_base(img_size=(config['aug_crop_res'],config['aug_crop_res']))
                if config['backbone_weights'] == 'in1k':
                    ckpt_path = '/remote_logging/checkpoints/timm/jx_vit_base_p16_224-80ecf9dd.pth'
                    #trunk.load_param(ckpt_path)
                    trunk._load_weights(ckpt_path)
                elif config['backbone_weights'] == 'pass':
                    ckpt_path = '/remote_logging/checkpoints/pass/pass_vit_base.pth'
                    trunk.load_param(ckpt_path)
                fpn_dim = 768
            else:
                raise Exception
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(VitHead(trunk, fpn_dim))
            else:
                fpn_dim = 384
            # Create backbone
            backbone = PassVitSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim)
        elif config['model'] == 'resnet':
            if config['backbone_arch'] == 'resnet50':
                trunk = torchvision.models.resnet50(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 1024
            else:
                raise Exception
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(ResnetHead(trunk))
            else:
                fpn_dim = 256
            # Create backbone
            backbone = ResnetSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim)
        elif config['model'] == 'lup_resnet':
            if config['backbone_arch'] == 'resnet50':
                trunk = lup_build_resnet_backbone()
                fpn_dim = 1024
            else:
                raise Exception
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(ResnetHead(trunk))
            else:
                fpn_dim = 256
            # Create backbone
            backbone = ResnetSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim)
        elif config['model'] == 'convnext':
            if config['backbone_arch'] == 'convnext_tiny':
                trunk = torchvision.models.convnext_tiny(weights=weights_backbone, progress=progress)
                fpn_dim = 384
            elif config['backbone_arch'] == 'convnext_small':
                trunk = torchvision.models.convnext_small(weights=weights_backbone, progress=progress)
                fpn_dim = None
            elif config['backbone_arch'] == 'convnext_base':
                trunk = torchvision.models.convnext_base(weights=weights_backbone, progress=progress)
                fpn_dim = 512
            elif config['backbone_arch'] == 'convnext_large':
                trunk = torchvision.models.convnext_large(weights=weights_backbone, progress=progress)
                fpn_dim = None
            else:
                raise Exception
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(ConvnextHead(trunk))
            else:
                fpn_dim = 256
            # Create backbone
            backbone = ConvnextSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim,
                freeze_non_norm=config['freeze_non_norm'],
                use_lora=config['use_lora'],
                merge_lora=config['merge_lora'])
        elif config['model'] == 'swin':
            if config['backbone_arch'] == 'swin_t':
                trunk = torchvision.models.swin_t(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 384 
            elif config['backbone_arch'] == 'swin_s':
                trunk = torchvision.models.swin_s(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 384 
            elif config['backbone_arch'] == 'swin_b':
                trunk = torchvision.models.swin_b(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 512
            elif config['backbone_arch'] == 'swin_v2_t':
                trunk = torchvision.models.swin_v2_t(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 384 
            elif config['backbone_arch'] == 'swin_v2_s':
                trunk = torchvision.models.swin_v2_s(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 384 
            elif config['backbone_arch'] == 'swin_v2_b':
                trunk = torchvision.models.swin_v2_b(
                    weights=weights_backbone, progress=progress)
                fpn_dim = 512
            else: raise NotImplementedError
            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(SwinHead(trunk))
            else:
                fpn_dim = 256
            # Create backbone
            backbone = SwinSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim,
                freeze_non_norm=config['freeze_non_norm'],
                use_lora=config['use_lora'],
                merge_lora=config['merge_lora'])
        elif config['model'] == 'solider_swin':
            semantic_weight = 0.6

            if config['backbone_weights'] == 'in1k':
                convert_weights = True
            else:
                convert_weights = False

            if config['backbone_arch'] == 'swin_t':
                solider_weight_path = '/remote_logging/checkpoints/solider_full/swin_tiny.pth'
                in1k_weight_path = '/remote_logging/checkpoints/timm/swin_tiny_in1k.pth'
                trunk = swin_tiny_patch4_window7_224(drop_path_rate=0.1,
                    semantic_weight=semantic_weight,
                    convert_weights=convert_weights)
                fpn_dim = 384
            elif config['backbone_arch'] == 'swin_s':
                solider_weight_path = '/remote_logging/checkpoints/solider_full/swin_small.pth'
                in1k_weight_path = '/remote_logging/checkpoints/timm/swin_small_in1k.pth'
                trunk = swin_small_patch4_window7_224(drop_path_rate=0.1,
                    semantic_weight=semantic_weight,
                    convert_weights=convert_weights)
                fpn_dim = 384
            elif config['backbone_arch'] == 'swin_b':
                #solider_weight_path = '/remote_logging/checkpoints/solider_teacher/swin_base_tea.pth'
                #in1k_weight_path = '/remote_logging/checkpoints/timm/swin_base_in1k.pth'
                trunk = swin_base_patch4_window7_224(drop_path_rate=0.1,
                    semantic_weight=semantic_weight,
                    convert_weights=convert_weights)
                fpn_dim = 512
            else: raise NotImplementedError
            print('==> Backbone weight initialization: {}'.format(
                config['backbone_weights']))
            if config['backbone_weights'] == 'random':
                pass 
            elif config['backbone_weights'] == 'solider':
                def _resume_from_ckpt(ckpt_path, model, convert=False):
                    if convert:
                        model.backbone.swin.convert_weights = True
                        model.backbone.swin.init_weights(ckpt_path)
                    else:
                        ckpt = torch.load(ckpt_path)
                        if 'state_dict' in ckpt.keys():
                            ckpt = ckpt['state_dict']

                        count = 0
                        miss = []
                        for ckpt_key in ckpt:
                            if 'backbone' in ckpt_key:
                                model_key1 = ckpt_key.replace('backbone.', '')
                                model.state_dict()[model_key1].copy_(ckpt[ckpt_key])
                                count += 1
                            else:
                                miss.append(ckpt_key)
                        print('%d loaded, %d missed:' %(count,len(miss)),miss)
                    return 0
                #_resume_from_ckpt(solider_weight_path, trunk)
                _resume_from_ckpt(config['init_ckpt_path'], trunk)
            elif config['backbone_weights'] == 'in1k':
                #trunk.init_weights(pretrained=in1k_weight_path)
                trunk.init_weights(pretrained=config['init_ckpt_path'])

            # Create conv5 embedding head
            if config['emb_align_mode'] == 'conv5':
                emb_head = copy.deepcopy(
                    SoliderSwinHead(trunk, out_channels=fpn_dim))
            else:
                fpn_dim = 256
            # Create backbone
            backbone = SoliderSwinSPBackbone(trunk,
                config['backbone_arch'],
                use_classifier=config['use_classifier_train'],
                freeze_backbone=config['freeze_backbone'],
                fpn_dim=fpn_dim,
                freeze_non_norm=config['freeze_non_norm'],
                use_lora=config['use_lora'],
                merge_lora=config['merge_lora'])

        anchor_generator = _default_anchorgen()

    elif config['neck_type'] == 'orig':
        # Backbone model
        if config['model'] == 'resnet':
            backbone, emb_head = build_resnet(arch=config['backbone_arch'],
                pretrained=config['pretrained'],
                freeze_backbone_batchnorm=config['freeze_backbone_batchnorm'],
                freeze_layer1=config['freeze_layer1'],
                freeze_backbone=config['freeze_backbone'])
        elif config['model'] == 'convnext':
            backbone, emb_head = build_convnext(arch=config['backbone_arch'],
                pretrained=config['pretrained'],
                freeze_layer1=config['freeze_layer1'],
                freeze_backbone=config['freeze_backbone'])
        else:
            raise NotImplementedError
        anchor_generator = _basic_anchorgen()
    else:
        raise NotImplementedError

    head = SPNetHead(
        config,
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes=2,
        norm_layer=partial(nn.GroupNorm, 32),
        oim_lut_size=oim_lut_size,
        emb_head=emb_head,
    )
    for h in head.regression_head.values():
        h._loss_type = "giou" # giou
    model = SPNet(config, backbone, num_classes, anchor_generator=anchor_generator, head=head, **kwargs)

    missing_keys = None
    if config['ckpt_path'] is not None:
        assert os.path.exists(config['ckpt_path'])
        print('==> Loading model checkpoint from: {}'.format(config['ckpt_path']))
        ckpt = torch.load(config['ckpt_path'], weights_only=True)
        _state_dict = ckpt['state_dict']
        state_dict = {}
        for k,v in _state_dict.items():
            state_dict[k[6:]] = v
        # If we don't want to delete any keys
        if config['test_only']:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            return model, missing_keys

        # MOCO copy teacher
        if config['moco_copy_teacher']:
            print('==> Copying moco teacher weights...')
            _copy_key_prefix(state_dict,
                'moco_backbone.', 'backbone.')
            _copy_key_prefix(state_dict,
                'head.moco_align_emb_head.', 'head.align_emb_head.')

        # Delete keys with potentially conflicting params
        ## Delete keys with specific prefix
        ### Make sure we aren't using MOCO or KP if we are deleting those keys
        assert not config['use_moco']
        assert not (config['classifier_objective'] == 'lwf')
        ### Helper function
        def _del_key_prefix(state_dict, prefix):
            key_list = list(state_dict.keys())
            for key in key_list:
                if key.startswith(prefix):
                    del state_dict[key]
        ### Delete keys
        _del_key_prefix(state_dict, 'moco_backbone.')
        _del_key_prefix(state_dict, 'head.moco_')
        _del_key_prefix(state_dict, 'lwf_backbone.')
        _del_key_prefix(state_dict, 'head.reid_loss.')
        _del_key_prefix(state_dict, 'backbone.classifier.')
        #### Added for SOLIDER backbone loading
        _del_key_prefix(state_dict, 'backbone.head.')
        _del_key_prefix(state_dict, 'backbone.norm.')

        # Delete extra swin keys
        if config['model'] == 'swin':
            _del_key_prefix(state_dict,
                'backbone.head')
            _del_key_prefix(state_dict,
                'backbone.norm')

        ## Delete bridge weights if loading from QC model
        ## - we want the default initialization for these weights
        if config['pretrain_train_mode'] == 'qc':
            _del_key_prefix(state_dict, 'head.bridge_layer')

        ## If loading from shared head model
        if config['ckpt_load_share']:
            ### Delete NAE head weights, which may not be the right size
            if True:
                print('WARNING:', "_del_key_prefix(state_dict, 'head.align_emb_head.nae_head.projectors.feat_res')")
                _del_key_prefix(state_dict, 'head.align_emb_head.nae_head.projectors.feat_res')
            ### Add loc head weights to state_dict, which are copy of reid
            ### emb_head
            prefix = 'head.align_emb_head.emb_head.'
            new_prefix = 'head.align_emb_loc_head.1.emb_head.'
            key_list = list(state_dict.keys())
            for key in key_list:
                if key.startswith(prefix):
                    suffix = key.split(prefix)[1]
                    new_key = new_prefix + suffix
                    value = state_dict[key]
                    state_dict[new_key] = value

            ### For GFN, use align_emb_head weights for GFN heads too
            if config['use_gfn']:
                prefix = 'head.align_emb_head.emb_head.'
                new_prefix_list = [
                    'head.gfn.query_reid_head.',
                    'head.gfn.head.image_reid_head.',
                ]
                key_list = list(state_dict.keys())
                for key in key_list:
                    if key.startswith(prefix):
                        suffix = key.split(prefix)[1]
                        for new_prefix in new_prefix_list:
                            new_key = new_prefix + suffix
                            value = state_dict[key]
                            state_dict[new_key] = value
        else:
            ### For GFN, use align_emb_head weights for GFN heads too
            if config['use_gfn']:
                ####
                prefix = 'head.align_emb_head.emb_head.'
                new_prefix_list = [
                    'head.gfn.query_reid_head.',
                    'head.gfn.head.image_reid_head.',
                ]
                key_list = list(state_dict.keys())
                for key in key_list:
                    if key.startswith(prefix):
                        suffix = key.split(prefix)[1]
                        for new_prefix in new_prefix_list:
                            new_key = new_prefix + suffix
                            value = state_dict[key]
                            state_dict[new_key] = value
                ####
                prefix = 'head.align_emb_head.nae_head.'
                new_prefix_list = [
                    'head.gfn.query_emb_head.',
                    'head.gfn.head.image_emb_head.',
                ]
                key_list = list(state_dict.keys())
                for key in key_list:
                    if key.startswith(prefix):
                        suffix = key.split(prefix)[1]
                        for new_prefix in new_prefix_list:
                            new_key = new_prefix + suffix
                            value = state_dict[key]
                            state_dict[new_key] = value

        ## Duplicate keys for cascade modules
        print('==> Checkpoint load cascade: {}'.format(config['ckpt_load_cascade']))
        if config['ckpt_load_cascade']:
            ### List of keys which need to be duplicated
            prefix_list = [
                ('head.classification_head', 0),
                ('head.regression_head', 0),
                ('head.bridge_layer', 0),
                ('head.align_emb_loc_head', 1),
            ] 
            key_list = list(state_dict.keys())
            for key in key_list:
                for prefix, start_idx in prefix_list:
                    if key.startswith(prefix):
                        suffix = key.split(prefix)[-1][3:]
                        assert key == f'{prefix}.{start_idx}.{suffix}', (key, f'{prefix}.{start_idx}.{suffix}')
                        value = state_dict[key]
                        for cascade_idx in range(start_idx+1, config['num_cascade_steps']+1):
                            new_key = f'{prefix}.{cascade_idx}.{suffix}'
                            state_dict[new_key] = value
        ## Load patched state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if config['ckpt_load_cascade'] and config['ckpt_load_share']:
            if not config['moco_copy_teacher']:
                pass
        elif config['ckpt_load_cascade']:
            pass

        assert len(unexpected_keys) == 0, unexpected_keys
        ## Handle LORA weights
        if config['use_lora'] and config['merge_lora']:
            print('==> Merging LORA weights...')
            lora_to_orig(model.backbone)
            model.requires_grad_(True)

    # Cleanup missing keys
    if missing_keys is not None:
        ## Remove MOCO keys
        missing_keys = [k for k in missing_keys if 'moco' not in k]
        ## Remove reid loss
        missing_keys = [k for k in missing_keys if 'reid_loss' not in k]
        ## Remove running mean and variance
        missing_keys = [k for k in missing_keys if 'running_mean' not in k]
        missing_keys = [k for k in missing_keys if 'running_var' not in k]

        #
        if False:
            from pprint import pprint
            print('Missing keys:')
            pprint(missing_keys)
            print('Unexpected keys:')
            pprint(unexpected_keys)
            exit()

        ## Add bridge_layer
        for k in state_dict:
            if ('bridge_layer' in k) and (k not in missing_keys):
                missing_keys.append(k)
    else:
        ## Add bridge_layer
        missing_keys = []
        for k in [n for n,p in model.named_parameters()]:
            if 'bridge_layer' in k:
                missing_keys.append(k)

    # Return model and missing keys
    return model, missing_keys
