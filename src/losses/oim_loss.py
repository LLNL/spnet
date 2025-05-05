# Global imports
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_metric_learning.losses import NTXentLoss, TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner, BatchEasyHardMiner, HDCMiner

class HDCMinerK(HDCMiner):
    def __init__(self, filter_k=10000, **kwargs):
        super().__init__(**kwargs)
        self.filter_k = filter_k
        self.add_to_recordable_attributes(
            list_of_names=["filter_k"], is_stat=False
        )
        self.reset_idx()

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        self.set_idx(labels, ref_labels)

        for name, (anchor, other) in {
            "pos": (self.a1, self.p),
            "neg": (self.a2, self.n),
        }.items():
            if len(anchor) > 0:
                pairs = mat[anchor, other]
                num_pairs = len(pairs)
                k = min(num_pairs, self.filter_k)
                largest = self.should_select_largest(name)
                _, idx = torch.topk(pairs, k=k, largest=largest)
                self.filter_original_indices(name, idx)

        return self.a1, self.p, self.a2, self.n

class MaskedBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x, m=None):
        if m is None:
            x = super().forward(x)
        else:
            #
            y = super().forward(x[m])
            x[m] = y
            #
            #self.eval()
            #z = super().forward(x[~m])
            #x[~m] = z
            #self.train()
        return x

# Refactored OIM loss with safe float16 computation
class OIMLossNorm(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.use_cq = use_cq

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if self.use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0

        # Norm layer
        #self.norm = nn.GroupNorm(num_groups=1, num_channels=num_features)
        #self.norm = nn.BatchNorm1d(num_features)
        self.norm = MaskedBatchNorm1d(num_features, momentum=1.0)

    def forward(self, inputs, label):
        # Normalize inputs
        if False:
            #inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)
            pass
        else:
            self.norm.eval()
            norm_inputs = self.norm(inputs.view(-1, self.num_features))
            self.norm.train()

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            if self.use_cq:
                bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        if False:
            outputs_labeled = torch.mm(
                F.normalize(inputs),
                F.normalize(self.lut.clone()).T
            )
        else:
            outputs_labeled = torch.mm(
                F.normalize(norm_inputs),
                F.normalize(self.norm(self.lut.clone(), m=~bad_lut_mask)).T
            )
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.use_cq:
            if False:
                outputs_unlabeled = torch.mm(
                    F.normalize(inputs),
                    F.normalize(self.cq.clone()).T
                )
            else:
                self.norm.eval()
                outputs_unlabeled = torch.mm(
                    F.normalize(norm_inputs),
                    F.normalize(self.norm(self.cq.clone(), m=~bad_cq_mask)).T
                )
                self.norm.train()
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        with torch.no_grad():
            targets = label
            for x, y in zip(inputs, targets):
                if y < len(self.lut):
                    #self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                    self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum) * x
                elif self.use_cq:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class OIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm = None

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None

    def forward(self, inputs, label):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)
        if False:
            new_label = torch.unique(label)
            new_input_list = []
            for y in torch.unique(new_label):
                x = inputs[label==y].mean(dim=0)
                new_input_list.append(x)
            new_inputs = torch.stack(new_input_list)
            inputs, label = new_inputs, new_label

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        with torch.no_grad():
            targets = label
            for x, y in zip(inputs, targets):
                if y < len(self.lut):
                    self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                elif self.cq is not None:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class OIMLossCQ(nn.Module):
    """
    Assumes CQ size is greater then unique pids in any possible batch.
    - Design goal is to make sure we always compare against <= |CQ| elements
    - Otherwise we could just append current batch to CQ or something similar
    """
    def __init__(self, num_features, num_cq_size, oim_scalar, oim_momentum,
            mode='ntx'):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_unlabeled = num_cq_size
        self.oim_scalar = oim_scalar
        self.oim_momentum = oim_momentum
        self.ignore_index = -1
        self.mode = mode

        # Setup loss
        if self.mode == 'triplet':
            self.loss_func = TripletMarginLoss(
                margin=0.1,
                swap=False,
                smooth_loss=False,
                triplets_per_anchor=10,
            )

        # Reset CQ
        self.reset_cq()

    def reset_cq(self):
        print('==> Resetting OIM CQ...')
        # Setup buffers
        self.register_buffer("emb_cq", torch.zeros(self.num_unlabeled, self.num_features))
        self.register_buffer("label_cq", 
            torch.full((self.num_unlabeled,), self.ignore_index, dtype=torch.long)
        )
        self.register_buffer("age_cq", 
            torch.zeros((self.num_unlabeled,), dtype=torch.long)
        )
        self.header_cq = 0

    def forward(self, inputs, labels, moco_inputs=None):
        # Compute instance means
        unique_loss_pid = torch.unique(labels)
        unique_loss_emb_list = []
        for pid in unique_loss_pid:
            pid_mask = labels == pid
            if moco_inputs is None:
                _unique_loss_emb = torch.mean(inputs[pid_mask], dim=0)
            else:
                _unique_loss_emb = torch.mean(moco_inputs[pid_mask], dim=0)
            unique_loss_emb_list.append(_unique_loss_emb)
        unique_loss_emb = torch.stack(unique_loss_emb_list)

        # XXX: use input mean as well
        if False:
            loss_emb_list = []
            for pid in unique_loss_pid:
                pid_mask = labels == pid
                _loss_emb = torch.mean(inputs[pid_mask], dim=0)
                loss_emb_list.append(_loss_emb)
            inputs = torch.stack(loss_emb_list)
            labels = unique_loss_pid

        # Normalize embeddings
        inputs = F.normalize(inputs, dim=1)
        unique_loss_emb = F.normalize(unique_loss_emb, dim=1) 

        # Compute CQ updates
        cm, cn = 0, 0
        with torch.no_grad():
            # Update age of each valid element by 1 iter
            self.age_cq[self.label_cq!=self.ignore_index] += 1
            # Iterate through unique emb, label pairs
            for x, y in zip(unique_loss_emb, unique_loss_pid):
                m = y == self.label_cq
                c = m.sum().item()
                assert c <= 1, "Make sure CQ labels are unique before update."
                if c == 1:
                    cm += 1
                    i = torch.where(m)[0]
                    assert self.label_cq[i] == y
                    
                    # Bump up the position of the embedding in the CQ
                    # - ensures all elements in batch will have match in the CQ
                    self.emb_cq[self.header_cq] = x
                    self.label_cq[self.header_cq] = y
                    self.age_cq[self.header_cq] = 1
                    # Ignore previous position if it isn't the current CQ index
                    if i != self.header_cq:
                        self.label_cq[i] = self.ignore_index
                        # Embedding ignored, set age to 0
                        self.age_cq[i] = 0
                else:
                    cn += 1
                    self.emb_cq[self.header_cq] = x
                    self.label_cq[self.header_cq] = y
                    # Embedding replaced, set age to 1
                    self.age_cq[self.header_cq] = 1
                # Update CQ index    
                self.header_cq = (self.header_cq + 1) % self.num_unlabeled

        # Max age
        #print('Match/Nonmatch: {}/{}'.format(cm, cn))
        #print('Min/max label: {}/{}'.format(labels.min(), labels.max()))

        # Mask of labels in CQ which are not set to ignore: use these entries for loss
        good_mask = self.label_cq != self.ignore_index
        #print('CQ fill: {}/{}'.format(good_mask.sum().item(), self.label_cq.numel()))
        #print('Max emb age:', self.age_cq[good_mask].max())

        # Make sure every input label has exactly one match in the CQ
        label_mask = labels.unsqueeze(1) == self.label_cq[good_mask].unsqueeze(0)
        assert torch.all(label_mask.sum(dim=1) == 1)

        # Generate labels for cross entropy loss
        xe_labels = torch.where(label_mask)[1]

        # Compute cosine similarity
        projected = torch.einsum('id,jd->ij', inputs, self.emb_cq[good_mask])

        if self.mode == 'ntx':
            # Multiply projections by (inverse) temperature scalar
            projected *= self.oim_scalar
            # Compute loss
            ## for numerical stability with float16, we divide before computing the sum to compute the mean
            ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
            _loss_oim = F.cross_entropy(projected, xe_labels, reduction='none')
            loss_oim = (_loss_oim / _loss_oim.size(0)).sum()
        elif self.mode == 'triplet':
            loss_oim = self.loss_func(inputs, labels,
                ref_emb=self.emb_cq[good_mask],
                ref_labels=self.label_cq[good_mask])
        else:
            raise NotImplementedError

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class OIMLossCQElem(nn.Module):
    """
    Assumes CQ size is greater then unique pids in any possible batch.
    - Design goal is to make sure we always compare against <= |CQ| elements
    - Otherwise we could just append current batch to CQ or something similar
    """
    def __init__(self, num_features, num_cq_size, oim_scalar, oim_momentum):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_unlabeled = num_cq_size
        self.oim_scalar = oim_scalar
        self.oim_momentum = oim_momentum
        self.ignore_index = -1

        self.loss_func = NTXentLoss(temperature=0.1)
        #self.loss_miner = HDCMinerK(filter_k=10000)
        self.loss_miner = BatchHardMiner()

        # Reset CQ
        self.reset_cq()

    def reset_cq(self):
        print('==> Resetting OIM CQ...')
        # Setup buffers
        self.register_buffer("emb_cq", torch.zeros(self.num_unlabeled, self.num_features))
        self.register_buffer("label_cq", 
            torch.full((self.num_unlabeled,), self.ignore_index, dtype=torch.long)
        )
        self.register_buffer("age_cq", 
            torch.zeros((self.num_unlabeled,), dtype=torch.long)
        )
        self.header_cq = 0

    def forward(self, inputs, labels, moco_inputs=None):
        # Normalize embeddings
        inputs = F.normalize(inputs, dim=1)
        moco_inputs = F.normalize(moco_inputs, dim=1)

        # Compute CQ updates
        cm, cn = 0, 0
        with torch.no_grad():
            # Update age of each valid element by 1 iter
            self.age_cq[self.label_cq!=self.ignore_index] += 1
            # Iterate through unique emb, label pairs
            for x, y in zip(moco_inputs, labels):
                self.emb_cq[self.header_cq] = x
                self.label_cq[self.header_cq] = y
                # Embedding replaced, set age to 1
                self.age_cq[self.header_cq] = 1
                # Update CQ index    
                self.header_cq = (self.header_cq + 1) % self.num_unlabeled

        # Max age
        print('Min/max label: {}/{}'.format(labels.min(), labels.max()))

        # Mask of labels in CQ which are not set to ignore: use these entries for loss
        good_mask = self.label_cq != self.ignore_index
        print('CQ fill: {}/{}'.format(good_mask.sum().item(), self.label_cq.numel()))
        print('Max emb age:', self.age_cq[good_mask].max())

        miner_output = self.loss_miner(inputs, labels,
            ref_emb=self.emb_cq[good_mask],
            ref_labels=self.label_cq[good_mask])

        loss_oim = self.loss_func(inputs, labels, miner_output,
            ref_emb=self.emb_cq[good_mask],
            ref_labels=self.label_cq[good_mask])

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class OIMLossIoU(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm = None

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None

    def forward(self, inputs, label, iou):
        if False:
            iou = torch.sigmoid(iou.logit()/0.5)
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)
        if False:
            new_label = torch.unique(label)
            new_input_list = []
            new_iou_list = []
            for y in torch.unique(new_label):
                x = F.normalize((inputs[label==y] * iou[label==y].unsqueeze(1)).mean(dim=0), dim=0)
                #x = F.normalize((inputs[label==y]).mean(dim=0), dim=0)
                new_input_list.append(x)
                new_iou = iou[label==y].mean()
                new_iou_list.append(new_iou)
            new_inputs = torch.stack(new_input_list)
            new_iou = torch.tensor(new_iou_list).to(iou)
            inputs, label = new_inputs, new_label
            #iou = new_iou
            iou = torch.ones_like(new_iou)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        prob_labels = torch.zeros_like(projected) 
        prob_labels.scatter_(1, label.unsqueeze(1), iou.unsqueeze(1).to(projected))
        ignore_mask = label != self.ignore_index
        _loss_oim = F.cross_entropy(projected[ignore_mask], prob_labels[ignore_mask], reduction='none') * iou[ignore_mask]
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        with torch.no_grad():
            targets = label
            for x, y, i in zip(inputs, targets, iou):
                if y < len(self.lut):
                    self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x * i, dim=0)
                elif self.cq is not None:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)

        # Return loss
        return loss_oim

# Refactored OIM loss with safe float16 computation
class OIMLossSafeNew(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super().__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.norm = None
        self.use_cq = use_cq

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None

    def forward(self, inputs, label):
        # Normalize inputs
        norm_inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = norm_inputs.mm(F.normalize(self.lut.t().clone()))
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = norm_inputs.mm(F.normalize(self.cq.t().clone()))
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        COMBINED = False
        NORM = True
        if COMBINED:
            with torch.no_grad():
                targets = label
                for y in torch.unique(targets):
                    y_mask = targets == y
                    if y < len(self.lut):
                        if NORM:
                            x = F.normalize(inputs[y_mask].mean(dim=0), dim=0)
                            self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                        else:
                            x = inputs[y_mask].mean(dim=0)
                            self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum) * x
                    elif self.use_cq:
                        x = inputs[y_mask].mean(dim=0)
                        self.cq[self.header_cq] = x
                        self.header_cq = (self.header_cq + 1) % self.cq.size(0)
        else:
            with torch.no_grad():
                targets = label
                for x, y in zip(inputs, targets):
                    if y < len(self.lut):
                        if NORM:
                            self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * F.normalize(x, dim=0), dim=0)
                        else:
                            self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum) * x
                    elif self.use_cq:
                        self.cq[self.header_cq] = x
                        self.header_cq = (self.header_cq + 1) % self.cq.size(0)


        # Return loss
        return loss_oim
