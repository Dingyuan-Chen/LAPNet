import argparse
import torch
import os

from avalanche.evaluation.metrics import (
    timing_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage

from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.training.supervised import (
    Naive, VAETraining
)

from data.GlobalDataset import GlobalDataset, train_transforms, test_transforms
from benchmark.naive_segmentation import SegmentationTemplate
from evaluation.segm_eval import make_segm_metrics

from benchmark.plugins import EWCPlugin, AGEMPlugin, LwFPlugin, GenerativeReplayPlugin
from avalanche.training.plugins import ReplayPlugin, SynapticIntelligencePlugin, TrainGeneratorAfterExpPlugin

from avalanche.training.storage_policy import (
    ParametricBuffer,
    Anchor_HerdingSelectionStrategy,
)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PolyLearningRate(object):
    def __init__(self,
                 optimizer,
                 base_lr,
                 power,
                 max_iters,
                 ):
        self._base_lr = base_lr
        self.power = power
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.last_epoch = 0

    @property
    def base_lr(self):
        return self._base_lr

    def step(self):
        factor = (1 - self.last_epoch / self.max_iters) ** self.power
        cur_lr = self.base_lr * factor
        self.last_epoch += 1
        set_lr(self.optimizer, cur_lr)

def load_dataset(ROOT, train=False, transforms=None):
    paths = {
        'masks': ROOT + '/mask',
        'images': ROOT + '/images'
    }

    fn_mapping = {'masks': lambda name: os.path.splitext(name)[0] + '.png'}
    dataset = GlobalDataset(paths, train=train, transforms=transforms, fn_mapping=fn_mapping)

    return dataset

def main(args):
    RNGManager.set_random_seeds(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying plugin: {args.plugins}...")
    print(f"Using device: {device}")

    train_mb_size = 16
    train_epochs = 25

    train_Global_1 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_1 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    train_Global_2 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_2 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    train_Global_3 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_3 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    train_Global_4 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_4 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    train_Global_5 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_5 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    train_Global_6 = load_dataset('/{path_to_files}/', train=True, transforms=train_transforms)
    test_Global_6 = load_dataset('/{path_to_files}/', train=False, transforms=test_transforms)

    benchmark = dataset_benchmark(
        [train_Global_1, train_Global_2, train_Global_3, train_Global_4, train_Global_5, train_Global_6],
        [test_Global_1, test_Global_2, test_Global_3, test_Global_4, test_Global_5, test_Global_6]
    )

    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory=f'{args.model_dir}/checkpoints/',
        ),
        map_location=device
    )

    load_strategy, _ = checkpoint_plugin.load_checkpoint_if_exists(cdy_last_exp=args.checkpoint_at)
    model = load_strategy.model

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0001
    )

    len_data = max([len(stream.dataset) for stream in benchmark.train_stream])
    lr_scheduler = PolyLearningRate(optimizer, base_lr=0.03, power=0.9,
                                    max_iters=(len_data * train_epochs) // train_mb_size + 1)

    plugins = [
        LRSchedulerPlugin(
            lr_scheduler,
            step_granularity="iteration",
            first_exp_only=False,
            first_epoch_only=False,
        ),
    ]

    if args.plugins == 'ewc':
        plugins.append(EWCPlugin(ewc_lambda=0.4, mode="separate"))
    elif args.plugins == 'si':
        plugins.append(SynapticIntelligencePlugin(si_lambda=0.0001))
    elif args.plugins == 'lwf':
        plugins.append(LwFPlugin(alpha=1, temperature=2))
    elif args.plugins == 'replay':
        plugins.append(ReplayPlugin(mem_size=600))
    elif args.plugins == 'agem':
        plugins.append(AGEMPlugin(patterns_per_experience=100, sample_size=16))
    elif args.plugins == 'dgr':
        generator = MlpVAE((3, 512, 512), nhid=2, device=device)
        generator = generator.to(device)

        lr = 0.01
        optimizer_generator = torch.optim.Adam(
            filter(lambda p: p.requires_grad, generator.parameters()),
            lr=lr,
            weight_decay=0.0001,
        )
        generator_strategy = VAETraining(
            model=generator,
            optimizer=optimizer_generator,
            criterion=VAE_loss,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=train_mb_size,
            device=device,
            plugins=[
                GenerativeReplayPlugin()
            ],
        )

        rp = GenerativeReplayPlugin(
            generator_strategy=generator_strategy,
        )

        tgp = TrainGeneratorAfterExpPlugin()

        plugins.append(tgp)
        plugins.append(rp)
    elif args.plugins == 'lapnet':
        storage_p = ParametricBuffer(
            max_size=600,
            groupby='experience',
            selection_strategy=Anchor_HerdingSelectionStrategy(model=model, layer_name='aspp_head.head.0.project')
        )
        plugins.append(ReplayPlugin(mem_size=600, storage_policy=storage_p))

    # Create loggers (as usual)
    loggers = [
        InteractiveLogger(),
    ]

    # Create the evaluation plugin (as usual)
    eval_plugin = EvaluationPlugin(
        make_segm_metrics(),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
    )

    # create strategy
    strategy = SegmentationTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=plugins,
        evaluator=eval_plugin,
    )

    strategy.eval(benchmark.test_stream[:], num_workers=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugins",
        type=str,
        default='naive',
        choices=["naive", "ewc", "si", "lwf", "replay", "agem", "dgr", "lapnet"],
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=''
    )
    parser.add_argument(
        "--checkpoint_at",
        type=int,
        default=-1
    )
    main(parser.parse_args())
