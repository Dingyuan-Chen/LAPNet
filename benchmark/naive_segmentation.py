from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from pkg_resources import parse_version
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils.data_loader import (
    detection_collate_fn,
    TaskBalancedDataLoader,
    detection_collate_mbatches_fn,
)
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.models import FeatureExtractorBackbone

from torch.nn import AdaptiveAvgPool2d


class SegmentationTemplate(SupervisedTemplate):
    """
    The object detection naive strategy.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine-tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.

    This strategy can be used as a template for any object detection strategy.
    This template assumes that the provided model follows the same interface
    of torchvision detection models.

    For more info, refer to "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
        scaler=None,
    ):
        """
        Creates a naive detection strategy instance.

        :param model: The PyTorch detection model. This strategy accepts model
            from the torchvision library (as well as all model sharing the same
            interface/behavior)
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param scaler: The scaler from PyTorch Automatic Mixed Precision
            package. More info here: https://pytorch.org/docs/stable/amp.html.
            Defaults to None.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        self.scaler = scaler  # torch.cuda.amp.autocast scaler
        """
        The scaler from PyTorch Automatic Mixed Precision package.
        More info here: https://pytorch.org/docs/stable/amp.html
        """

        # Object Detection attributes
        self.detection_loss_dict = None
        """
        A dictionary of detection losses.

        Only valid during the training phase.
        """

        self.detection_predictions = None
        """
        A list of detection predictions.

        This is different from mb_output: mb_output is a list of dictionaries 
        (one dictionary for each image in the input minibatch), 
        while this field, which is populated after calling `criterion()`,
        will be a dictionary {image_id: list_of_predictions}.

        Only valid during the evaluation phase. 
        """
        self.opt_experience = 0
        self.feature_extractor = FeatureExtractorBackbone(model, 'aspp_head.head.0.project')
        self.final_pool = AdaptiveAvgPool2d(output_size=(1, 1))

    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        **kwargs
    ):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            Used only if `PyTorch >= 1.7.0`.
        """

        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers

        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=detection_collate_fn,
            **other_dataloader_args
        )

    def make_eval_dataloader(self, num_workers=0, pin_memory=True, **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            collate_fn=detection_collate_fn,
        )

    def criterion(self):
        """
        Compute the loss function.

        The initial loss dictionary must be obtained by first running the
        forward pass (the model will return the detection_loss_dict).
        This function will only obtain a single value.

        Beware that the loss can only be obtained for the training phase as no
        loss dictionary is returned when evaluating.
        """
        if self.is_training:
            self.detection_loss_dict = self.segm_loss(self.mb_output, self.mb_y)
            return sum(loss for loss in self.detection_loss_dict.values())
        else:

            self.detection_predictions = {
                'targets': self.mb_y,
                'outputs': self.mb_output.sigmoid(),
            }
            return torch.zeros((1,))

    def segm_loss(self, y_pred, y_true):
        loss_dict = dict()

        is_prior_learning = True
        y_pred, layout_feas = y_pred

        if is_prior_learning and self.experience.current_experience > 0:
            with torch.no_grad():
                layout_feas = y_true.shape[0] * layout_feas / sum(layout_feas)
                weight = torch.zeros(y_true.shape).to(y_true.device)
                for i in range(y_true.shape[0]):
                    weight[i] = layout_feas[i]
                weight = weight.view(-1)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        if is_prior_learning and self.experience.current_experience > 0:
            loss_dict['bce_loss'] = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), weight=weight)
            loss_dict['soft_dice_loss'] = dice_loss_with_logits(y_pred, y_true, ignore_index=255)
        else:
            loss_dict['bce_loss'] = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
            loss_dict['soft_dice_loss'] = dice_loss_with_logits(y_pred, y_true, ignore_index=255)
        return loss_dict

    def forward(self):
        """
        Compute the model's output given the current mini-batch.

        For the training phase, a loss dictionary will be returned.
        For the evaluation phase, this will return the model predictions.
        """
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logit = self.model(self.mb_x)
                features = self.feature_extractor(self.mb_x)

            return [logit, features]
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logit = self.model(self.mb_x)

            return logit

    def _unpack_minibatch(self):
        # Unpack minibatch mainly takes care of moving tensors to devices.
        # In addition, it will prepare the targets in the proper dict format.
        images = torch.cat(list(im.unsqueeze(0) for im in self.mbatch[0]), dim=0).to(self.device)

        targets = torch.cat(list(mask.unsqueeze(0) for mask in self.mbatch[1]), dim=0).to(self.device)
        self.mbatch = [images, targets, torch.as_tensor(self.mbatch[2]).to(self.device)]

    def backward(self):
        if self.scaler is not None:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def optimizer_step(self, **kwargs):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

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
