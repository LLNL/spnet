# Global imports
## general imports
import os
import math
import copy
import numpy as np
import collections
## torch and torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
# typing
from typing import Dict, Tuple, List, Any, Optional
from torch import Tensor

# Package imports
## Models
from osr.models.backbone import build_resnet, build_convnext
from osr.models.transform import GeneralizedRCNNTransform
from osr.models.gfn import GalleryFilterNetwork, SafeBatchNorm1d
## Losses
from osr.losses.oim_loss import OIMLossSafe, OIMLossNorm, OIMLossCQ


# Fork of torchvision RPN with safe fp16 loss
class SafeRegionProposalNetwork(RegionProposalNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, objectness: Tensor, pred_bbox_deltas: Tensor,
            labels: List[Tensor], regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1 / 9,
                reduction="none",
            )
            / (sampled_inds.numel())
        ).sum()

        _objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], reduction='none')
        objectness_loss = (_objectness_loss / _objectness_loss.size(0)).sum()

        return objectness_loss, box_loss

def get_seqnext(config, oim_lut_size):
    # Instantiate SeqNeXt model
    model = SeqNeXt(config, oim_lut_size=oim_lut_size)

    # Load model weights if available
    if config['ckpt_path'] is not None:
        assert os.path.exists(config['ckpt_path'])
        print('==> Loading model checkpoint from: {}'.format(config['ckpt_path']))
        ckpt = torch.load(config['ckpt_path'])
        _state_dict = ckpt['state_dict']
        state_dict = {}
        for k,v in _state_dict.items():
            state_dict[k[6:]] = v
        # Delete keys with potentially conflicting params
        ## Helper to delete keys from torch module state dict
        def _del_key(state_dict, key):
            if key in state_dict:
                del state_dict[key]
        _del_key(state_dict, 'roi_heads.reid_loss.lut')
        _del_key(state_dict, 'roi_heads.reid_loss.cq')
        _del_key(state_dict, 'roi_heads.reid_loss.emb_cq')
        _del_key(state_dict, 'roi_heads.reid_loss.label_cq')
        _del_key(state_dict, 'roi_heads.reid_loss.age_cq')
        #
        model.load_state_dict(state_dict, strict=False)#.get_state_dict(progress=progress))
        if config['use_lora'] and config['merge_lora']:
            print('==> Merging LORA weights...')
            lora_to_orig(model.backbone)
            model.requires_grad_(True)

        # Copy MOCO teacher weights to student for finetune
        if config['moco_copy_teacher'] and (not config['test_only']):
            print('==> Copying MOCO teacher weights to student...')
            ## backbone
            del model.backbone
            model.backbone = copy.deepcopy(model.moco_backbone)
            del model.moco_backbone
            ## emb head
            del model.head.align_emb_head
            model.head.align_emb_head = copy.deepcopy(model.head.moco_align_emb_head)
            del model.head.moco_align_emb_head
            ##
            model.requires_grad_(True)
    return model

# SeqNeXt module
class SeqNeXt(nn.Module):
    def __init__(self, config, oim_lut_size=None, device='cpu'):
        super().__init__()

        #
        self.device = device

        #
        self.use_gfn = config['use_gfn']
        self.gfn_use_image_lut = config['gfn_use_image_lut']

        # MOCO
        self.use_moco = config['use_moco']

        # Backbone model
        if config['model'] == 'resnet':
            backbone, prop_head = build_resnet(arch=config['backbone_arch'],
                pretrained=config['pretrained'],
                freeze_backbone_batchnorm=config['freeze_backbone_batchnorm'], freeze_layer1=config['freeze_layer1'])
        elif config['model'] == 'convnext':
            backbone, prop_head = build_convnext(arch=config['backbone_arch'],
                pretrained=config['pretrained'],
                freeze_layer1=config['freeze_layer1'])
        else:
            raise NotImplementedError

        # RPN Anchor settings
        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 1.0, 2.0),)
        rpn_conv_depth = 1

        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios,
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            conv_depth=rpn_conv_depth,
        )
        pre_nms_top_n = dict(
            training=config['rpn_pre_nms_topn_train'], testing=config['rpn_pre_nms_topn_test']
        )
        post_nms_top_n = dict(
            training=config['rpn_post_nms_topn_train'], testing=config['rpn_post_nms_topn_test']
        )
        rpn = SafeRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=config['rpn_pos_thresh_train'],
            bg_iou_thresh=config['rpn_neg_thresh_train'],
            batch_size_per_image=config['rpn_batch_size_train'],
            positive_fraction=config['rpn_pos_frac_train'],
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=config['rpn_nms_thresh'],
        )

        # Set number of channel inputs for box predictors
        box_channels = prop_head.out_channels[-1]

        # Set up box predictors
        faster_rcnn_predictor = FastRCNNPredictor(box_channels, 2)
        reid_head = copy.deepcopy(prop_head)

        # Set up RoI Align
        box_roi_pool = reid_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_head = reid_head

        # Re-ID
        if config['reid_objective'] == 'cq':
            reid_loss = OIMLossCQ(config['emb_dim'], config['oim_cq_size'],
                config['oim_scalar'], config['oim_momentum'])
            reid_loss.norm = None
        elif config['reid_objective'] == 'oim':
            if config['emb_norm_type'] == 'lutnorm':
                reid_loss = OIMLossNorm(config['emb_dim'], oim_lut_size,
                    config['oim_cq_size'], config['oim_momentum'], config['oim_scalar'])
            else:
                reid_loss = OIMLossSafe(config['emb_dim'], oim_lut_size,
                    config['oim_cq_size'], config['oim_momentum'], config['oim_scalar'])

        # Embedding head
        featmap_names = reid_head.featmap_names
        in_channels = reid_head.out_channels
        embedding_head = NormAwareEmbedding(
            featmap_names=featmap_names, in_channels=in_channels, dim=config['emb_dim'],
                norm_type=config['emb_norm_type'], lut_norm=reid_loss.norm)
        if config['box_head_mode'] == 'rcnn':
            embedding_head.rescaler = None
            box_predictor = lambda x: None
        else:
            box_predictor = BBoxRegressor(box_channels, num_classes=2)

        # Gallery-Filter Network
        if self.use_gfn:
            ## Build Gallery Filter Network
            self.gfn = GalleryFilterNetwork(reid_roi_pool, reid_head,
                embedding_head, mode=config['gfn_mode'],
                gfn_activation_mode=config['gfn_activation_mode'],
                emb_dim=config['emb_dim'], temp=config['gfn_train_temp'], se_temp=config['gfn_se_temp'],
                filter_neg=config['gfn_filter_neg'],
                use_image_lut=config['gfn_use_image_lut'],
                gfn_query_mode=config['gfn_query_mode'],
                gfn_scene_pool_size=config['gfn_scene_pool_size'],
                gfn_norm_type=config['emb_norm_type'],
                pos_num_sample=config['gfn_num_sample'][0], neg_num_sample=config['gfn_num_sample'][1],
                reid_loss=reid_loss)
        else:
            self.gfn = None

        # RoI Heads
        roi_heads = SeqRoIHeads(
            # Re-ID criterion
            reid_loss=reid_loss,
            # OIM
            num_pids=oim_lut_size,
            num_cq_size=config['oim_cq_size'],
            oim_momentum=config['oim_momentum'],
            oim_scalar=config['oim_scalar'],
            # SeqNeXt
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            det_score=config['det_score'],
            cws_score=config['cws_score'],
            # Norm Layer
            norm_type=config['emb_norm_type'],
            #
            gfn=self.gfn,
            gfn_use_image_lut=self.gfn_use_image_lut,
            embedding_head=embedding_head,
            # parent class
            box_roi_pool=box_roi_pool,
            reid_roi_pool=reid_roi_pool,
            prop_head=prop_head,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=config['roi_head_pos_thresh_train'],
            bg_iou_thresh=config['roi_head_neg_thresh_train'],
            batch_size_per_image=config['roi_head_batch_size_train'],
            positive_fraction=config['roi_head_pos_frac_train'],
            bbox_reg_weights=None,
            score_thresh=config['roi_head_score_thresh_test'],
            nms_thresh=config['roi_head_nms_thresh_test'],
            detections_per_img=config['roi_head_detections_per_image_test'],
            emb_dim=config['emb_dim'],
            box_head_mode=config['box_head_mode'],
            box_channels=box_channels,
            use_moco=config['use_moco'],
            moco_copy_teacher=config['moco_copy_teacher'],
        )

        # batch and pad images
        transform = GeneralizedRCNNTransform()

        self.backbone = backbone
        if self.use_moco:
            self.moco_backbone = copy.deepcopy(backbone)
            self.moco_backbone.requires_grad_(False)
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = config['lw_rpn_reg']
        self.lw_rpn_cls = config['lw_rpn_cls']
        self.lw_proposal_reg = config['lw_proposal_reg']
        self.lw_proposal_cls = config['lw_proposal_cls']
        self.lw_box_reg = config['lw_box_reg']
        self.lw_box_cls = config['lw_box_cls']
        self.lw_box_reid = config['lw_box_reid']


    def inference(self, images: List[Tensor], targets:Optional[List[Dict[str, Tensor]]]=None, inference_mode:str='both') -> List[Dict[str, Tensor]]:
        #
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        num_images = len(original_image_sizes)
        images, targets = self.transform(images, targets)
        bb_features = self.backbone(images.tensors)

        # Get image features from the GFN
        if self.use_gfn:
            scene_emb = self.gfn.get_scene_emb(bb_features).split(1, 0)
        else:
            scene_emb = [torch.empty(0) for _ in range(num_images)]

        #
        reid_features = bb_features

        detections = [{} for _ in range(num_images)]
        embeddings = [torch.empty(0) for _ in range(num_images)]
        if (inference_mode in ('gt', 'both')) and (targets is not None):
            # query
            boxes = [t["boxes"] for t in targets]
            section_lens = [len(b) for b in boxes]
            box_features = self.roi_heads.reid_roi_pool(reid_features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features)
            _embeddings, _ = self.roi_heads.embedding_head(box_features)
            embeddings = _embeddings.split(section_lens, dim=0)
        if (inference_mode in ('det', 'both')) or (inference_mode in ('gt', 'both')):
            # gallery
            rpn_features = bb_features
            proposals, _ = self.rpn(images, rpn_features, targets)
            detections, _, _ = self.roi_heads(
                bb_features, proposals, images.image_sizes, targets
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )

        # Reorganize outputs into single list of dict
        output_list = [{
            'det_boxes': d['boxes'],
            'det_scores': d['scores'],
            'det_labels': d['labels'],
            'det_emb': d['embeddings'],
            'gt_emb': e,
            'gt_loc_emb': e,
            #'scene_emb': s,
        } for d, e, s in zip(detections, embeddings, scene_emb)]

        # Return output
        return output_list


    def forward(self, images: List[Tensor], targets:Optional[List[Dict[str, Tensor]]]=None, inference_mode:str='both') -> Dict[str, Tensor]:
        if not self.training:
            return self.inference(images, targets, inference_mode=inference_mode)

        images, targets = self.transform(images, targets)
        bb_features = self.backbone(images.tensors)
        if self.use_moco:
            moco_bb_features = self.moco_backbone(images.tensors)
        else:
            moco_bb_features = None
        
        # GFN training
        losses = {}
        if self.use_gfn and self.training:
            gfn_losses, _ = self.gfn(bb_features, targets, images.image_sizes)
            losses.update(gfn_losses)

        # RPN
        rpn_features = bb_features
        proposals, proposal_losses = self.rpn(images, rpn_features, targets)
        _, detector_losses, fork_features = self.roi_heads(bb_features, proposals, images.image_sizes, targets, moco_bb_features=moco_bb_features)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        # store rpn and rcnn losses
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights, (GFN lw=1 implicitly)
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["sim_loss"] *= self.lw_box_reid # (1/10)
        return losses, fork_features


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        reid_loss,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        prop_head,
        reid_head,
        reid_roi_pool=None,
        norm_type='batchnorm',
        emb_dim=256,
        gfn=None,
        embedding_head=None,
        gfn_use_image_lut=False,
        box_head_mode='nae',
        box_channels=None,
        det_score='scs',
        cws_score='scs',
        use_moco=False,
        moco_copy_teacher=False,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)

        self.det_score = det_score
        self.cws_score = cws_score
        self.reid_loss = reid_loss
        self.embedding_head = embedding_head
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        self.reid_roi_pool = reid_roi_pool
        self.prop_head = prop_head
        # rename the method inherited from parent class
        #self.postprocess_proposals = self.postprocess_detections
        self.num_pids = num_pids
        self.emb_dim = emb_dim
        self.gfn = gfn
        self.gfn_use_image_lut = gfn_use_image_lut
        self.box_head_mode = box_head_mode
        self.use_moco = use_moco
        self.moco_copy_teacher = moco_copy_teacher

        if self.box_head_mode == 'rcnn':
            self.score_predictor = FastRCNNPredictor(box_channels, 1)

        # MOCO
        if self.use_moco or self.moco_copy_teacher:
            self.moco_box_head = copy.deepcopy(self.box_head)
            self.moco_box_head.requires_grad_(False)
            self.moco_embedding_head = copy.deepcopy(self.embedding_head)
            self.moco_embedding_head.requires_grad_(False)

    def forward(self, bb_features: Dict[str, Tensor], proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]], targets:Optional[List[Dict[str, Tensor]]]=None, moco_bb_features=None) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            proposal_pid_labels = []
            proposal_reg_targets = [torch.empty(0)]

        reg_proposals = proposals
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
        else:
            proposal_labels = [torch.empty(0)]

        # ------------------- Faster R-CNN head ------------------ #
        prop_features = bb_features
        proposal_features = self.box_roi_pool(prop_features, reg_proposals, image_shapes)
        proposal_features = self.prop_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features[self.prop_head.featmap_names[-1]]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, reg_proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, box_img_ids, box_pid_labels, box_reg_targets = self.select_training_samples(boxes, targets)
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_detections(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )
            box_pid_labels = [torch.empty(0)]
            box_reg_targets = [torch.empty(0)]

        cws = True
        gt_det = None

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            device = boxes[0].device
            boxes = [torch.zeros(0, 4).to(device)]
            labels = [torch.zeros(0).to(device)]
            scores = [torch.zeros(0).to(device)]
            embeddings = [torch.zeros(0, self.emb_dim).to(device)]
            return [dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores), embeddings=torch.cat(embeddings))], {}, None
        else:
            scores = [torch.empty(0)]

        # --------------------- Baseline head -------------------- #
        ###
        reg_boxes = boxes
        if self.training:
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
        else:
            box_labels = [torch.empty(0)]

        ###
        fork_features = self.box_roi_pool(bb_features, reg_boxes, image_shapes)
        reid_features = box_features = self.box_head(fork_features)
        box_embeddings, box_cls_scores = self.embedding_head(reid_features)
        if self.box_head_mode == 'rcnn':
            box_cls_scores, box_regs = self.score_predictor(box_features[self.box_head.featmap_names[-1]])
            box_cls_scores = box_cls_scores.squeeze(1)
            box_regs = box_regs.repeat(1, 2)
        else:
            #box_regs = self.box_predictor(box_features[self.box_head.featmap_names[-1]])
            raise Exception

        if box_cls_scores.dim() == 0:
            box_cls_scores = box_cls_scores.unsqueeze(0)

        result, losses = [{}], {}
        if self.training:
            # Detection losses
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )

            # Reid loss
            _box_pid_labels = torch.cat(box_pid_labels)
            fg_mask = _box_pid_labels > 0 
            fg_box_embeddings = box_embeddings[fg_mask]

            ## GFN
            query_labels = _box_pid_labels[fg_mask]-1
            query_feats = fg_box_embeddings
            ## MOCO
            if self.use_moco:
                moco_fork_features = self.box_roi_pool(moco_bb_features, reg_boxes, image_shapes)
                moco_reid_features = self.moco_box_head(moco_fork_features)
                moco_box_embeddings, _ = self.moco_embedding_head(moco_reid_features)
                fg_moco_box_embeddings = moco_box_embeddings[fg_mask]
                ## Compute re-id loss
                sim_loss = self.reid_loss(query_feats, query_labels, moco_inputs=fg_moco_box_embeddings)
            else:
                ## Compute re-id loss
                sim_loss = self.reid_loss(query_feats, query_labels)
            losses['sim_loss'] = sim_loss
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            result = [{} for i in range(num_images)]
            for i in range(num_images):
                result[i] = dict(
                    boxes=boxes[i], labels=labels[i],
                    scores=scores[i], embeddings=embeddings[i]
                )
        return result, losses, fork_features

    def get_boxes(self, box_regression: Tensor, proposals: List[Tensor], image_shapes: List[Tuple[int, int]]) -> List[Tensor]:
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        embeddings: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        fcs:List[Tensor],
        gt_det:Optional[Dict[str, Tensor]]=None,
        cws:bool=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # Which stage to use for detector scores
        ## First Classification Score (FCS)
        fcs_scores = fcs[0]
        ## Second Classification Score (SCS)
        scs_scores = torch.sigmoid(class_logits)

        # Which stage to use for detector scores
        if self.det_score == 'fcs':
            # First Classification Score (FCS)
            pred_scores = fcs_scores
        elif self.det_score == 'scs':
            # Second Classification Score (SCS)
            pred_scores = scs_scores
        else:
            raise Exception

        # Which stage to use for detector scores
        if self.cws_score == 'fcs':
            # First Classification Score (FCS)
            cws_scores = fcs_scores.view(-1, 1)
        elif self.cws_score == 'scs':
            # Second Classification Score (SCS)
            cws_scores = scs_scores.view(-1, 1)
        elif self.cws_score == 'none':
            # Second Classification Score (SCS)
            cws_scores = torch.ones_like(pred_scores).view(-1, 1)
        else:
            raise Exception

        # Normalize embeddings
        if True:
            embeddings = F.normalize(embeddings)

        # Whether to weight embeddings by the detector scores
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * cws_scores

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

class ChannelWiseNoise(nn.Module):
    def __init__(self, d, epsilon=1e-5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, d))
        if True:
            self.logvar = nn.Parameter(2*torch.log(torch.eye(d)))
        else:
            self.logvar = nn.Parameter(2*torch.log(torch.ones(1, d)))
        self.epsilon = epsilon

    def sample(self):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * self.logvar)
        if self.training:
            if True:
                d = torch.distributions.MultivariateNormal(self.mu, std)
            else:
                d = torch.distributions.Normal(self.mu, std)
            x = d.sample((10,))
            sample_mean = x.mean(dim=0)
            sample_std = x.std(dim=0)
            return sample_mean, sample_std
        else:
            if True:
                return self.mu, std.diag()
            else:
                return self.mu, std

    def forward(self, x):
        mu, sigma = self.sample()
        return (x - mu) / torch.sqrt(sigma**2 + self.epsilon)

class ChannelWiseNoise2(nn.Module):
    def __init__(self, d, epsilon=1e-5):
        super().__init__()
        self.mean_mu = nn.Parameter(torch.zeros(1, d))
        self.mean_logvar = nn.Parameter(2*torch.log(torch.ones(1, d)*epsilon))
        self.std_mu = nn.Parameter(torch.ones(1, d))
        self.std_logvar = nn.Parameter(2*torch.log(torch.ones(1, d)*epsilon))
        self.epsilon = epsilon

    def sample(self):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        mean_std = torch.exp(0.5 * self.mean_logvar)
        std_std = torch.exp(0.5 * self.std_logvar)
        if self.training:
            mean_d = torch.distributions.Normal(self.mean_mu, mean_std)
            std_d = torch.distributions.Normal(self.std_mu, std_std)
            return mean_d.sample(), std_d.sample()
        else:
            return self.mean_mu, self.std_mu

    def forward(self, x):
        mu, sigma = self.sample()
        return (x - mu) / torch.sqrt(sigma**2 + self.epsilon)

class ChannelWiseNoise3(nn.Module):
    def __init__(self, d, epsilon=1e-5):
        super().__init__()
        self.std = torch.ones(1, d) * 0.01
        self.epsilon = epsilon

    def sample(self, x):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if self.training:
            x_hat = torch.distributions.Normal(x, self.std.repeat(x.shape[0], 1).to(x))
            return x_hat.sample()
        else:
            return x

    def forward(self, x):
        return self.sample(x)

class ChannelWiseAffine(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, d))
        self.beta = nn.Parameter(torch.zeros(1, d))

    def forward(self, x):
        return x*self.gamma + self.beta

class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.

    For SeqNeXt, we do not use embedding norms as scores by default, but retain the option
    mainly for comparitive purposes.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256, norm_type='batchnorm', lut_norm=None):
        super().__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim
        self.lut_norm = lut_norm

        if norm_type == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_type == 'batchnorm':
            norm_layer = SafeBatchNorm1d
        elif norm_type == 'groupnorm':
            norm_layer = nn.GroupNorm
        elif norm_type == 'lutnorm':
            norm_layer = None

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            if norm_type in (None, 'lutnorm'):
                proj = nn.Linear(in_channel, indv_dim)
                init.normal_(proj.weight, std=0.01)
                init.constant_(proj.bias, 0)
            else:
                if norm_type == 'batchnorm':
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim),
                        norm_layer(indv_dim, affine=False))
                elif norm_type == 'groupnorm':
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim), norm_layer(num_groups=32, num_channels=indv_dim))
                else:
                    proj = nn.Sequential(nn.Linear(in_channel, indv_dim), norm_layer(indv_dim))
                init.normal_(proj[0].weight, std=0.01)
                init.constant_(proj[0].bias, 0)
                #init.normal_(proj[1].weight, std=0.01)
                #init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        # Affine True by default for both BatchNorm1d, LayerNorm
        if norm_type == None:
            self.rescaler = nn.Identity()
        elif norm_type == 'lutnorm':
            self.rescaler = nn.Identity()
        elif norm_type == 'groupnorm':
            self.rescaler = nn.Identity()
        else:
            self.rescaler = nn.Identity()#norm_layer(1)

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
            norms = embeddings.norm(2, 1, keepdim=True)
            if False:
                embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            ## Needed for dtype=half right now
            embeddings = embeddings.to(norms)
            if self.lut_norm:
                if self.training:
                    self.lut_norm.eval()
                    embeddings = self.lut_norm(embeddings)
                    self.lut_norm.train()
                else:
                    embeddings = self.lut_norm(embeddings)
            if self.rescaler is None:
                norms = norms.squeeze()
            else:
                norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for fk, fv in featmaps.items():
                fv = self._flatten_fc_input(fv)
                ## Loop for torch.jit
                for pk, pv in self.projectors.items():
                    if pk == fk:
                        outputs.append(pv(fv))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            if False:
                embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            ## Needed for dtype=half right now
            embeddings = embeddings.to(norms)
            if self.lut_norm:
                if self.training:
                    self.lut_norm.eval()
                    embeddings = self.lut_norm(embeddings)
                    self.lut_norm.train()
                else:
                    embeddings = self.lut_norm(embeddings)
            if self.rescaler is None:
                norms = norms.squeeze()
            else:
                norms = self.rescaler(norms).squeeze()
            return embeddings, norms

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


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
        init.normal_(self.bbox_pred.weight, std=0.01)
        init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    proposal_cls_scores: Tensor,
    proposal_regs: Tensor,
    proposal_labels: List[Tensor],
    proposal_reg_targets: List[Tensor],
    box_cls_scores: Tensor,
    box_regs: Tensor,
    box_labels: List[Tensor],
    box_reg_targets: List[Tensor],
) -> Dict[str, Tensor]:
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    _loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels, reduction='none')
    loss_proposal_cls = (_loss_proposal_cls / _loss_proposal_cls.size(0)).sum()
    _loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float(), reduction='none')
    loss_box_cls = (_loss_box_cls / _loss_box_cls.size(0)).sum()

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="none",
    )
    loss_proposal_reg = (loss_proposal_reg / proposal_labels.numel()).sum()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="none",
    )
    loss_box_reg = (loss_box_reg / box_labels.numel()).sum()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )
