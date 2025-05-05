# Cert monkey patch for torchvision model download (if needed)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Global imports
import argparse
import os
import sys
import copy
import shutil
import glob
from collections import OrderedDict
import collections
from pprint import pprint
from enum import IntEnum, auto
## torch
import torch
import torch.nn.functional as F
## lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, BaseFinetuning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy

# Package imports
## engine
from osr.engine import evaluate
from osr.engine import utils as engine_utils
from osr.models.seqnext import get_seqnext
from osr.models.spnet import spnet


class EvalStage(IntEnum):
    QUERY_CENTRIC1 = 0
    QUERY_CENTRIC2 = 1
    LOSS = 2
    CLASSIFIER = 3
    OBJECT_CENTRIC = 4

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

class PLModule(LightningModule):
    def __init__(self, config=None, checkpoint_dir=None, lr=None):
        super().__init__()
        ###
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.compute_anchor_metrics = config['compute_anchor_metrics']

        # Make training (more) reproducible
        ## We do not control for non-determinism in pytorch, torchvision
        if config['use_random_seed']:
            ## Random seeds
            pl.seed_everything(config['random_seed'])

        #
        self.search_mode = config['search_mode']

        # setup
        self.train_loader, num_train_pid = engine_utils.get_train_loader(self.config,
            rank=self.global_rank, world_size=self.config['world_size'],
            partition='train')

        # dictionary of test loaders
        self.test_loader_dict = {k:[] for k in EvalStage}
        num_test_pid = 0
        ## test loader for reid mode
        if self.config['test_eval_mode'] == 'reid':
            test_loader = engine_utils.get_test_loader(self.config)
            self.test_loader_dict[EvalStage.QUERY_CENTRIC1] = test_loader
        ## test loader for search mode XXX: warning, no classifier mode
        if self.config['test_eval_mode'] in ('search', 'all'):
            test_loader = engine_utils.get_test_loader(self.config)
            self.test_loader_dict[EvalStage.QUERY_CENTRIC1] = test_loader
            self.test_loader_dict[EvalStage.QUERY_CENTRIC2] = copy.deepcopy(test_loader)
        ## test loader for loss mode
        if self.config['test_eval_mode'] in ('loss', 'all'):
            ### modify some parameters in the config
            loss_config = copy.deepcopy(self.config)
            if self.config['test_eval_aug'] == 'test':
                loss_config['use_ssl'] = False
                loss_config['match_mode'] = 'other'
                loss_config['sampler_mode'] = 'pair'
                loss_config['aug_mode'] = 'wrs'
                loss_config['batch_size'] = 6
            ### build the train loader with the modified config
            loss_test_loader, num_test_pid = engine_utils.get_train_loader(loss_config,
                rank=self.global_rank, world_size=loss_config['world_size'],
                partition='test')
            self.test_loader_dict[EvalStage.LOSS] = loss_test_loader
            ### set epoch
            if self.config['test_eval_aug'] == 'train':
                test_index_list = loss_test_loader.batch_sampler.set_epoch(0) 
                loss_test_loader.dataset.set_epoch(0, test_index_list) 

        ## test loader for loss mode
        if self.config['test_eval_mode'] == 'detect':
            test_loader = engine_utils.get_test_loader(self.config)
            self.test_loader_dict[EvalStage.OBJECT_CENTRIC] = test_loader

        # Build model
        print("Creating model")
        if self.config['ps_model'] == 'seqnext':
            self.model = get_seqnext(self.config, oim_lut_size=num_train_pid)
        elif self.config['ps_model'] == 'spnet':
            self.model, self.uninitialized_keys = spnet(
                self.config, oim_lut_size=(num_train_pid, num_test_pid))
            if self.uninitialized_keys is None:
                self.remaining_keys = None
            else:
                self.remaining_keys = [n for n, p in self.model.named_parameters() if (p.requires_grad and (n not in self.uninitialized_keys))]

        # load eval protocol
        if not (self.config['test_eval_mode'] == 'loss'):
            protocol_list = evaluate.get_protocol_list(test_loader)
            assert len(protocol_list) == 1
            self.protocol = protocol_list[0]

        # Get list of modules needed for MOCO update
        if self.config['use_moco']:
            if self.config['ps_model'] == 'spnet':
                self.orig_module_list = [self.model.backbone, self.model.head.align_emb_head]
                self.moco_module_list = [self.model.moco_backbone, self.model.head.moco_align_emb_head]
            elif self.config['ps_model'] == 'seqnext':
                self.orig_module_list = [self.model.backbone, self.model.roi_heads.box_head, self.model.roi_heads.embedding_head]
                self.moco_module_list = [self.model.moco_backbone, self.model.roi_heads.moco_box_head, self.model.roi_heads.moco_embedding_head]

        # Set schedule
        if self.config['warmup_schedule'] is not None:
            self.schedule = 'warmup'
        else:
            self.schedule = 'regular'

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader_dict

    def test_dataloader(self):
        return self.test_loader_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, eval_stage=None):
        return self.on_validation_epoch_end(eval_stage=eval_stage)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        if batch_idx == 0:
            self.validation_step_outputs = []
        eval_stage = dataloader_idx
        #print('===')
        #print('EvalStage step:', eval_stage)
        #print('===')
        dataloader = self.val_dataloader()[dataloader_idx]
        if eval_stage == EvalStage.LOSS:
            # Set model to train mode
            self.model.train()
            # Set batchnorm layers to eval mode so we don't update running stats
            self.model.apply(set_bn_eval)
            # Set reid loss to eval mode
            self.model.head.reid_loss.eval()
            # Run regular training step
            output = self.training_step(batch, batch_idx, partition='Val')
        elif eval_stage == EvalStage.CLASSIFIER:
            output = evaluate.run_step_classifier(self.model, batch)
        elif eval_stage == EvalStage.OBJECT_CENTRIC:
            output = evaluate.run_step(self.model, batch,
                dataloader.sampler.query_id_list)
        elif eval_stage == EvalStage.QUERY_CENTRIC1:
            output = evaluate.run_step_query(self.model, batch,
                dataloader.sampler.query_id_list)
        elif eval_stage == EvalStage.QUERY_CENTRIC2:
            prev_dataloader = self.val_dataloader()[EvalStage.QUERY_CENTRIC1]
            output = evaluate.run_step_search(self.model, batch,
                prev_dataloader.sampler.query_id_list,
                prev_dataloader.query_lookup,
                prev_dataloader.image_lookup,
                self.protocol, search_mode=self.search_mode,
                compute_anchor_metrics=self.compute_anchor_metrics)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self, eval_stage=None):
        # Get outputs
        outputs = self.validation_step_outputs

        # Get eval stage
        #print('===')
        #print('EvalStage end:', eval_stage)
        #print('===')
        try:
            dataloader = self.val_dataloader()[eval_stage]
        except KeyError:
            print('key "{}" not in loader dict'.format(eval_stage))
            raise

        # Unpack outputs
        if eval_stage == EvalStage.CLASSIFIER:
            # Combine outputs
            class_logits_list, class_labels_list = list(zip(*outputs))
            class_logits = torch.cat(class_logits_list)
            class_labels = torch.cat(class_labels_list)
        elif eval_stage == EvalStage.OBJECT_CENTRIC:
            query_list, image_list, detection_list = list(zip(*outputs))
            image_lookup = {k: v for d in image_list for k, v in d.items()}
            query_lookup = {k: v for d in query_list for k, v in d.items()}
            detection_lookup = {k: v for d in detection_list for k, v in d.items()}
        elif eval_stage == EvalStage.QUERY_CENTRIC1:
            query_list, image_list = list(zip(*outputs))
            query_lookup = {k: v for d in query_list for k, v in d.items()}
            image_lookup = {k: v for d in image_list for k, v in d.items()}
            dataloader.query_lookup = query_lookup
            dataloader.image_lookup = image_lookup
        elif eval_stage == EvalStage.QUERY_CENTRIC2:
            prev_dataloader = self.val_dataloader()[EvalStage.QUERY_CENTRIC1]
            detection_list = outputs
            detection_lookup = {k: v for d in detection_list for k, v in d.items()}
            query_lookup = prev_dataloader.query_lookup
            image_lookup = prev_dataloader.image_lookup

        # Compute metrics
        if eval_stage == EvalStage.CLASSIFIER:
            metric_dict = evaluate.compute_metrics_classifier(class_logits, class_labels) 
            # Log results
            if not self.config['test_only']:
                self.log_dict({f'Val/{k}':v for k,v in metric_dict.items()}, add_dataloader_idx=False)
            else:
                print(flush=True)
                pprint(metric_dict)
        elif (eval_stage == EvalStage.QUERY_CENTRIC1) and (self.config['test_eval_mode'] == 'reid'):
            metric_dict = evaluate.compute_metrics_reid(self.model,
                dataloader,
                query_lookup, image_lookup,
                use_amp=self.config['use_amp'],
            )
            # Log results
            if not self.config['test_only']:
                self.log_dict({f'Val/{k}':v for k,v in metric_dict.items()}, add_dataloader_idx=False)
            else:
                print(flush=True)
                pprint(metric_dict)

        elif eval_stage in (EvalStage.OBJECT_CENTRIC, EvalStage.QUERY_CENTRIC2):
            # Compute metrics
            if eval_stage == EvalStage.OBJECT_CENTRIC:
                eval_mode = 'oc'
            elif eval_stage == EvalStage.QUERY_CENTRIC2:
                eval_mode = 'qc'
            metric_dict, value_dict, scores_dict = evaluate.compute_metrics(self.model,
                dataloader,
                query_lookup, image_lookup, detection_lookup,
                use_amp=self.config['use_amp'], use_gfn=self.config['use_gfn'],
                gfn_mode=self.config['gfn_mode'],
                eval_mode=eval_mode,
                compute_anchor_metrics=self.compute_anchor_metrics,
                use_cws=self.config['use_cws'],
            )
            # Log results
            if not self.config['test_only']:
                self.log_dict({f'Val/{k}':v for k,v in metric_dict.items()}, add_dataloader_idx=False)
                # Log score histograms
                for k, v in scores_dict.items():
                    if len(v) > 0:
                        self.logger.experiment.add_histogram(k, torch.tensor(v), self.global_step)
            else:
                print(flush=True)
                pprint(metric_dict)

    def configure_optimizers(self):
        print('==> Calling configure optimizers!')
        # Configure optimization schedule
        if self.schedule == 'warmup':
            print('==> Running WARMUP schedule...')
            config = self.config['warmup_schedule']
        elif self.schedule == 'regular':
            print('==> Running REGULAR schedule...')
            config = self.config['regular_schedule']

        # Get trainable params
        if self.schedule == 'warmup':
            params = [p for n, p in self.model.named_parameters() if (p.requires_grad and n in self.uninitialized_keys)]
            #print(params)
            #param_names = [n for n, p in self.model.named_parameters() if (p.requires_grad and n in self.uninitialized_keys)]
            #print(param_names)
            #exit()
        elif self.schedule == 'regular':
            if self.config['freeze_backbone']:
                params = [p for n,p in self.model.named_parameters() if (p.requires_grad and not n.startswith('backbone.body'))]
            else:
                if self.config['backbone_lr'] is not None:
                    backbone_params = [p for n,p in self.model.named_parameters() if (p.requires_grad and n.startswith('backbone.body'))]
                    other_params = [p for n,p in self.model.named_parameters() if (p.requires_grad and not n.startswith('backbone.body'))]
                    params = [
                        {
                            'name': 'backbone_params',
                            'params': backbone_params,
                            'lr': self.config['backbone_lr'],
                            'weight_decay': self.config['backbone_wd'],
                        },
                        {
                            'name': 'other_params',
                            'params': other_params,
                            'lr': config['lr'],
                            'weight_decay': config['wd'],
                        },
                    ]
                else:
                    params = [p for p in self.model.parameters() if p.requires_grad]
        else: raise NotImplementedError

        # Print param count info
        print('Num used param groups: {}/{}'.format(len(params), len(list(self.model.parameters()))))
        if True:
            def sizeof_number(number, currency=None):
                """
                format values per thousands : K-thousands, M-millions, B-billions. 
                
                parameters:
                -----------
                number is the number you want to format
                currency is the prefix that is displayed if provided (€, $, £...)
                
                """
                currency='' if currency is None else currency + ' '
                for unit in ['','K','M']:
                    if abs(number) < 1000.0:
                        return f"{number:6.1f}{unit}"
                    number /= 1000.0
                return f"{currency}{number:6.1f}B"

            if self.config['ps_model'] == 'spnet':
                group_name_dict = {
                    'backbone.body': 0,
                    'backbone.fpn': 0,
                    'head.align_emb_head.emb_head': 0,
                    'head.align_emb_loc_head.1.emb_head': 0,
                    'head.align_emb_loc_head.2.emb_head': 0,
                    'other': 0,
                }
            elif self.config['ps_model'] == 'seqnext':
                group_name_dict = {
                    'backbone.body': 0,
                    'roi_heads.prop_head': 0,
                    'roi_heads.box_head': 0,
                    'other': 0,
                }

            tot_params = 0
            print('===')
            for n,p in self.model.named_parameters():
                tot_params += p.numel()
                #if p.numel() > 1000000:
                #    print(n, sizeof_number(p.numel()))
                for name in group_name_dict:
                    if n.startswith(name):
                        group_name_dict[name] += p.numel()
                        break
                else:
                    group_name_dict['other'] += p.numel()
                    #print('other:', n, sizeof_number(p.numel()))
            print('===')
            for k,v in group_name_dict.items():
                print('{}: {}'.format(k, sizeof_number(v)))
            print('===')
            print('Tot params: {}'.format(sizeof_number(tot_params)))
            

        # Setup optimizer
        if config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=config['lr'], momentum=0.9,
                weight_decay=config['wd'])
        elif config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, lr=config['lr'],
                weight_decay=config['wd'])
        elif config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=config['lr'],
                weight_decay=config['wd'])
        else:
            raise Exception

        # Get schedule configs
        if self.config['warmup_schedule'] is None:
            schedule_config_list = [self.config['regular_schedule']]
        elif self.config['regular_schedule'] is None:
            schedule_config_list = [self.config['warmup_schedule']]
        else:
            schedule_config_list = [
                self.config['warmup_schedule'],
                self.config['regular_schedule']
            ]

        # Build list of schedulers
        scheduler_list = []
        milestone_list = []
        for config in schedule_config_list:
            _scheduler_list = []
            ## Setup warmup
            if config['use_warmup']:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(self.train_dataloader()) - 1)
                _scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )
                scheduler_list.append(_scheduler)
                if len(milestone_list) > 0:
                    m_prev = milestone_list[-1]
                else:
                    m_prev = 0
                milestone_list.append(warmup_iters + m_prev)
                T_sub = warmup_iters
            else:
                T_sub = 0

            ## Setup scheduler
            if config['scheduler'] == 'multistep':
                milestones = [x*len(self.train_dataloader()) for x in config['lr_steps']]
                _scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=milestones, gamma=config['lr_gamma'])
                scheduler_list.append(_scheduler)
            elif config['scheduler'] == 'cosine':
                T_max = len(self.train_dataloader())*config['epochs'] - T_sub
                _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max)
                if len(milestone_list) > 0:
                    m_prev = milestone_list[-1]
                else:
                    m_prev = 0
                milestone_list.append(T_max + m_prev)
                scheduler_list.append(_scheduler)

        # Build final combined scheduler object using list of schedules
        if len(scheduler_list) == 0:
            _scheduler = None
        elif len(scheduler_list) == 1:
            _scheduler = scheduler_list[0]
        else:
            _scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=scheduler_list,
                milestones=milestone_list[:-1],
            )

        # Package scheduler for lightnign
        scheduler = {
            "scheduler": _scheduler,
            "interval": "step",
            "frequency": 1,
        }

        # Return optimizer, scheduler
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        current_epoch = self.current_epoch
        # Set batch sampler for next epoch
        if self.config['use_ssl']:
            print('==> Setting train sampler and dataset for epoch: {}'.format(current_epoch))
            index_list = self.train_loader.batch_sampler.set_epoch(current_epoch) 
            self.train_loader.dataset.set_epoch(current_epoch, index_list) 
        elif self.config['sampler_mode'] in ('repeat', 'pair'):
            self.train_loader.batch_sampler.set_epoch(current_epoch) 

    def on_test_epoch_start(self):
        current_epoch = self.current_epoch
        # Reset OIM CQ
        if self.config['oim_cq_epoch_reset']:
            self.model.head.reid_loss.test_reid_loss.reset_cq()
            self.model.head.reid_loss.test_reid_loss.to(self.device)
        # Set batch sampler for next epoch
        if self.config['test_eval_mode'] in ('loss', 'all'):
            if self.config['use_ssl'] and (self.config['test_eval_aug'] == 'train'):
                loss_test_loader = self.test_loader_dict[EvalStage.LOSS]
                print('==> Setting test sampler and dataset for epoch: {}'.format(current_epoch))
                test_index_list = loss_test_loader.batch_sampler.set_epoch(current_epoch) 
                loss_test_loader.dataset.set_epoch(current_epoch, test_index_list) 
            elif self.config['sampler_mode'] in ('repeat', 'pair'):
                loss_test_loader = self.test_loader_dict[EvalStage.LOSS]
                loss_test_loader.batch_sampler.set_epoch(current_epoch) 
        #print('len(dataset):', len(self.test_loader_dict[EvalStage.LOSS].dataset))
        #print('len(dataloader):', len(self.test_loader_dict[EvalStage.LOSS]))
        #exit()

    def on_validation_epoch_start(self):
        self.on_test_epoch_start()

    def training_step(self, batch, batch_idx, partition='Train'):
        # Unpack batch
        images, targets = batch

        # Run batch through model
        loss_dict, _ = self.model(images, targets)

        # MOCO update if needed
        if self.config['use_moco'] and self.training:
            m = self.config['moco_momentum']
            # Perform MOCO parameter update
            with torch.no_grad():
                for orig_module, moco_module in zip(self.orig_module_list, self.moco_module_list):
                    for param_q, param_k in zip(orig_module.parameters(), moco_module.parameters()):
                        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

        # Accumulate and log losses
        losses = sum(loss_dict.values())
        if not self.config['test_only']:
            self.log_dict({f'{partition}/{k}':v.item() for k,v in loss_dict.items()}, add_dataloader_idx=False)
        return losses

from pytorch_lightning.trainer import call
from collections import ChainMap, defaultdict, OrderedDict
class QueryCentricValLoop(pl.loops._EvaluationLoop):
    def on_run_end(self, eval_stage):
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        # if `done` returned True before any iterations were done, this won't have been called in `on_advance_end`
        self.trainer._logger_connector.epoch_end_reached()
        self.trainer._logger_connector._evaluation_epoch_end()

        # hook
        self._on_evaluation_epoch_end(eval_stage)

        logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory
        # include any logged outputs on epoch_end
        epoch_end_logged_outputs = self.trainer._logger_connector.update_eval_epoch_metrics()
        all_logged_outputs = dict(ChainMap(*logged_outputs))  # list[dict] -> dict
        all_logged_outputs.update(epoch_end_logged_outputs)
        for dl_outputs in logged_outputs:
            dl_outputs.update(epoch_end_logged_outputs)

        # log metrics
        self.trainer._logger_connector.log_eval_end_metrics(all_logged_outputs)

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        if self.verbose and self.trainer.is_global_zero:
            self._print_results(logged_outputs, self._stage)

        return logged_outputs

    def _on_evaluation_epoch_end(self, eval_stage) -> None:
        """Runs ``on_{validation/test}_epoch_end`` hook."""
        trainer = self.trainer

        hook_name = "on_test_epoch_end" if trainer.testing else "on_validation_epoch_end"
        call._call_callback_hooks(trainer, hook_name)
        call._call_lightning_module_hook(trainer, hook_name, eval_stage=eval_stage)

        trainer._logger_connector.on_epoch_end()

    @torch.no_grad()
    def run(self):
        self._stage = 0
        self.setup_data()
        if self.skip:
            return []
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        prev_eval_stage = None
        assert data_fetcher is not None
        while True:
            try:
                batch, batch_idx, curr_eval_stage = next(data_fetcher)
                if prev_eval_stage is None:
                    prev_eval_stage = curr_eval_stage
                elif prev_eval_stage != curr_eval_stage:
                    self._store_dataloader_outputs()
                    self.on_run_end(prev_eval_stage)
                prev_eval_stage = curr_eval_stage
                self.batch_progress.is_last_batch = data_fetcher.done
                # run step hooks
                self._evaluation_step(batch, batch_idx, curr_eval_stage)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
                break
            finally:
                self._restarting = False
        self._store_dataloader_outputs()
        self.on_run_end(prev_eval_stage)
        return []

class FullFinetuneCallback(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        remaining_params = [p for n, p in pl_module.model.named_parameters() if (p.requires_grad and n in pl_module.remaining_keys)]
        for p in remaining_params:
            p.requires_grad = False

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            ## Build parameter group with remaining keys
            remaining_params = [p for n,p in pl_module.model.named_parameters() if n in pl_module.remaining_keys]
            for p in remaining_params:
                p.requires_grad = True
            lr = pl_module.config['regular_schedule']['lr']
            if pl_module.config['regular_schedule']['use_warmup']:
                lr = lr * 1e-3
            optimizer.add_param_group({
                'params': remaining_params, 
                'lr': lr,
            })


Args = collections.namedtuple('Args',
    ['default_config', 'trial_config', 'test', 'resume'],
    defaults=['./configs/default.yaml', './configs/default.yaml',
        False, False],
)

# Main function
def main(parse_config=True, **kwargs):
    # Parse args
    if parse_config:
        parser = argparse.ArgumentParser()
        parser.add_argument('--default_config', default='./configs/default.yaml')
        parser.add_argument('--trial_config', default='./configs/default.yaml')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--test_config', default=None)
        parser.add_argument('--resume', action='store_true')
        args = parser.parse_args()
    else:
        args = Args(**kwargs)

    # Load config
    default_config, tuple_key_list = engine_utils.load_config(args.default_config)
    trial_config, _ = engine_utils.load_config(args.trial_config, tuple_key_list=tuple_key_list)
    config = {**default_config, **trial_config}

    # For finetuning from pretraining, overwrite some args
    if config['pretrain_dir'] is not None:
        ## Load checkpoint using pretrain_dir and pretrain_epoch
        pretrain_dir = config['pretrain_dir']
        pretrain_epoch = config['pretrain_epoch']
        glob_path = f'{pretrain_dir}/**/*epoch={pretrain_epoch}*.ckpt'
        ckpt_list = glob.glob(glob_path, recursive=True)
        assert len(ckpt_list) == 1, glob_path
        config['ckpt_path'] = ckpt_list[0]
        assert os.path.exists(config['ckpt_path'])
        print('==> Loading model checkpoint found at: {}'.format(config['ckpt_path']))
        ## Build trial name from pretrain trial and given trial name
        pretrain_config_path = os.path.join(config['pretrain_dir'], 'config.yaml')
        pretrain_config, _ = engine_utils.load_config(pretrain_config_path,
            tuple_key_list=[])
        pretrain_trial_name = pretrain_config['trial_name']
        pretrain_epoch = config['pretrain_epoch']
        base_trial_name = config['trial_name']
        finetune_trial_name = f'{pretrain_trial_name}.ft-e{pretrain_epoch}:{base_trial_name}'
        config['trial_name'] = finetune_trial_name
        ## Copy pretrain config to dir
        copy_trial_dir = os.path.join(config['log_dir'],
            config['trial_name'])
        if not os.path.exists(copy_trial_dir):
            os.makedirs(copy_trial_dir)
        copy_pretrain_config = os.path.join(copy_trial_dir, 'pretrain_config.yaml')
        shutil.copy(pretrain_config_path, copy_pretrain_config)
        ## Store pretrain train_mode
        config['pretrain_train_mode'] = pretrain_config['train_mode']

    # For test mode, overwrite some args        
    if args.test:
        print('==> Test mode')
        ## Test only mode
        config['test_only'] = True
        ## Use separate test config if supplied
        if args.test_config is not None:
            test_config, _ = engine_utils.load_config(args.test_config, tuple_key_list)
            config.update(test_config)
        ## Find checkpoint path
        if (args.test_config is not None) and (test_config['ckpt_path'] is not None):
            print('Using checkpoint at: {}'.format(test_config['ckpt_path']))
        else:
            log_dir = os.path.join(config['log_dir'], config['trial_name'])
            ckpt_list = glob.glob(log_dir+'/**/*.ckpt', recursive=True)
            config['ckpt_path'] = ckpt_list[-1]
            print('Found checkpoint at: {}'.format(config['ckpt_path']))
    else:
        print('==> Train mode')

    # Additional global config operations
    ## For A100 GPUs, following line helps utilize Tensor Cores
    if True:
        torch.set_float32_matmul_precision('medium')

    # Initialize lightning module
    print('==> START init')
    model = PLModule(config)
    print('==> END init')

    # Initialize checkpoint callbacks
    ## Checkpoint every n epochs
    checkpoint_callback = ModelCheckpoint(
        monitor='step',
        filename=config['trial_name']+'_{epoch}_{step}',
        every_n_epochs=config['ckpt_interval'],
        save_top_k=-1,
        verbose=True,
        save_on_train_epoch_end=True,
    )
    ## Checkpoint every n iters:
    ## - keep latest checkpoint to resume in case of crash
    ckpt_dirpath = os.path.join(config['log_dir'], config['trial_name'])
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dirpath,
        monitor='step',
        filename='backup',
        every_n_train_steps=config['latest_ckpt_interval'],
        save_top_k=2,
        mode='max',
        verbose=True,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    # Set device
    strategy_device = "cuda:0"
    trainer_device = "auto"

    # Setup distributed params
    if config['distributed']:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = SingleDeviceStrategy(device=strategy_device)

    # Meta learning
    gradient_clip_val = 10.0

    # Set interval at which val set is checked
    eval_interval = config['eval_interval']
    if 0.0 < eval_interval < 1.0:
        val_check_interval = eval_interval
        check_val_every_n_epoch = 1
    elif eval_interval >= 1.0:
        val_check_interval = None
        check_val_every_n_epoch = eval_interval
    else:
        raise Exception('Invalid eval_interval: {}'.format(eval_interval))

    # Setup callback
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(),
        checkpoint_callback,
        latest_checkpoint_callback,
    ]
    if config['warmup_schedule'] is not None:
        callbacks.append(FullFinetuneCallback(
            unfreeze_at_epoch=config['warmup_schedule']['epochs']))

    # Get num warmup epochs
    if config['warmup_schedule'] is None:
        warmup_epochs = 0
    else:
        warmup_epochs = config['warmup_schedule']['epochs']

    # Get num warmup epochs
    if config['regular_schedule'] is None:
        regular_epochs = 0
    else:
        regular_epochs = config['regular_schedule']['epochs']

    # Set logger
    if config['test_only']:
        logger = None
    else:
        logger = TensorBoardLogger(
            save_dir=config['log_dir'],
            name=config['trial_name'],
            version='logs',
        )

    # Setup trainer
    trainer = Trainer(
        accelerator=trainer_device,
        devices=config['world_size'] if torch.cuda.is_available() else None,
        max_epochs=warmup_epochs+regular_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=logger,
        num_sanity_val_steps=0,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision="16-mixed",
        gradient_clip_val=gradient_clip_val,
        enable_checkpointing=config['ckpt_interval']>0,
        strategy=strategy,
        use_distributed_sampler=False,
    )

    # Set custom validation loop
    trainer.fit_loop.epoch_loop.val_loop = QueryCentricValLoop(trainer)
    trainer.test_loop = QueryCentricValLoop(trainer)

    #model.scaler = trainer.precision_plugin.scaler
    if config['test_only']:
        # Test model
        trainer.test(model)
    else:
        # Copy configs to log dir
        copy_trial_dir = os.path.join(config['log_dir'],
            config['trial_name'])
        if not os.path.exists(copy_trial_dir):
            os.makedirs(copy_trial_dir)
        copy_trial_config = os.path.join(copy_trial_dir, 'config.yaml')
        ## Copy trial config
        shutil.copy(args.trial_config, copy_trial_config)
        ## Copy default config
        shutil.copy(args.default_config, copy_trial_dir)

        # Fit model
        if args.resume:
            print('==> Resume training...')
            ## Find checkpoint path
            log_dir = os.path.join(config['log_dir'], config['trial_name'])
            ckpt_path = os.path.join(log_dir, 'last.ckpt')
            if os.path.exists(ckpt_path):
                print('Found checkpoint at: {}'.format(ckpt_path))
            else:
                print('No checkpoint found in: {}'.format(log_dir))
                sys.exit(1)
            ## Fit with checkpoint path
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            if config['eval_start']:
                ## Test model once before training
                trainer.test(model)
            print('==> START fit')
            ## Fit regular schedule
            trainer.fit(model)


# Run as module
if __name__ == '__main__':
    main()
