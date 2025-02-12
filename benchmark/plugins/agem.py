import warnings
import random
from typing import List
import torch

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.data_loader import (
    GroupBalancedInfiniteDataLoader,
)
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

import torch.nn.functional as F

class AGEMPlugin(SupervisedPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.sample_size = int(sample_size)

        self.buffers: List[
            make_classification_dataset
        ] = []  # one AvalancheDataset for
        # each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None
        self.reference_gradients = None

    def segm_loss(self, y_pred, y_true):
        loss_dict = dict()

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        loss_dict['bce_loss'] = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
        loss_dict['soft_dice_loss'] = dice_loss_with_logits(y_pred, y_true, ignore_index=255)

        return loss_dict

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            # loss = strategy._criterion(out, yref)
            loss_dict = self.segm_loss(out, yref)
            loss = sum(_ for _ in loss_dict.values())

            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = (
                    current_gradients - self.reference_gradients * alpha2
                )

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(
                            grad_proj[count : count + n_param].view_as(p)
                        )
                    count += n_param

    def after_training_exp(self, strategy, **kwargs):
        """Update replay memory with patterns from current experience."""
        self.update_memory(strategy.experience.dataset, **kwargs)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(iter(self.buffer_dliter))

    @torch.no_grad()
    def update_memory(self, dataset, num_workers=0, **kwargs):
        """
        Update replay memory with patterns from current experience.
        """
        if num_workers > 0:
            warnings.warn(
                "Num workers > 0 is known to cause heavy" "slowdowns in AGEM."
            )
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.subset(indices[:self.patterns_per_experience])

        self.buffers.append(dataset)

        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=(self.sample_size // len(self.buffers)),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers,
        )

        self.buffer_dliter = self.buffer_dataloader

def dice_coeff(y_pred, y_true, smooth_value: float = 1.0):
    inter = torch.sum(y_pred * y_true)
    z = y_pred.sum() + y_true.sum() + smooth_value
    return (2 * inter + smooth_value) / z

@torch.jit.script
def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, smooth_value: float = 1.0,
                          ignore_index: int = 255):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    mask = y_true == ignore_index
    valid = ~mask
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return 1. - dice_coeff(y_pred.sigmoid(), y_true, smooth_value)
