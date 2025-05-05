# Configs

This directory contains the config files needed to reproduce all experiments from the paper. We also specify which tables and plots in the paper correspond to the experiments.

## inference

SPNet-S and SPNet-L ConvNeXt backbone models, for which we provide pre-trained weights.

## baseline (SPNet-S)

Pre-train and fine-tune configs for all baseline experiments.

### baseline / pretrain / finetune
- Table 4, Table 5, Table 7, Table 8, Table 9, Figure 8: baseline result in several tables (SPNet-S with ConvNeXt-T), also denoted c0-share-d128

### cascade
- Figure 8: c2-sep-d2048 PT (SPNet-L with ConvNeXt-T backbone / 512 image size)

### norm
- Table 9: BatchNorm Logits vs. FixedNorm Logits

### weight_loading
- Figure 8: Comparison of architecture and weight loading hyperparameters

### qc
- Table 8: Comparison of QC fine-tuning to OC fine-tuning

## qc_oc

Experiments comparing query-centric (QC) with object-centric (OC) pre-training.

### opt_pretrain and opt_finetune
- Figure 7: comparison of layer optimization methods

### reid_pretrain and reid_finetune
- Table 6: comparison of re-id pre-training methods

## sample_ablation

Ablation of percent annotated samples for COCOPersons dataset.
- Figure 6: comparison of QC vs. OC as % annotations reduced / increased

## sota (SPNet-L)

Pre-train and fine-tune configs for the final best model.
- Table 4, Table 3 (top): comparison to SOTA models

## comparison

Comparison of different backbone-only initialization strategies
vs. our method, all using Swin-B backbone (SOLIDER version).
- Table 3 (bottom)

### baseline

Random initialization vs. IN1k, vs. SOLIDER LUPerson.

### pretrain

Pre-training using our approach with QC / OC on COCOPersons.

### ours_finetune

Fine-tuning from our pre-trained models.

### solider_finetune

Fine-tuning from SOLIDER pre-trained models, including our new SOLIDER COCO pre-trained models.

## benchmark

Inference speed for SPNet-L with ResNet50 backbone vs. other models w/ same backbone.
- Table 1
