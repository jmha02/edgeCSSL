import os
from pprint import pprint
import types

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from cassle.args.setup import parse_args_pretrain
from cassle.methods import METHODS
from cassle.distillers import DISTILLERS

try:
    from cassle.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from cassle.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from cassle.utils.checkpointer import Checkpointer
from cassle.utils.classification_dataloader import prepare_data as prepare_data_classification
from cassle.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
    split_dataset,
)


def main():
    seed_everything(5)

    args = parse_args_pretrain()

    # Remove devices parameter to avoid conflict with PyTorch Lightning
    if hasattr(args, 'devices'):
        delattr(args, 'devices')
        
    # Manually fix gpus parameter to use device 0 instead of device 1
    if hasattr(args, 'gpus') and args.gpus == [1]:
        args.gpus = [0]

    # online eval dataset reloads when task dataset is over
    args.multiple_trainloader_mode = "min_size"

    # set online eval batch size and num workers
    args.online_eval_batch_size = int(args.batch_size) if args.dataset == "cifar100" else None

    # split classes into tasks
    tasks = None
    if args.split_strategy == "class":
        assert args.num_classes % args.num_tasks == 0
        tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

    # pretrain and online eval dataloaders
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.unique_augs == 1

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            # imagenet or custom dataset
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, num_crops=[args.num_crops, args.num_small_crops]
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"

            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        train_dataset, online_eval_dataset = prepare_datasets(
            args.dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )

        task_dataset, tasks = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=args.task_idx,
            num_tasks=args.num_tasks,
            split_strategy=args.split_strategy,
        )

        task_loader = prepare_dataloader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loaders = {f"task{args.task_idx}": task_loader}

        if args.online_eval_batch_size:
            online_eval_loader = prepare_dataloader(
                online_eval_dataset,
                batch_size=args.online_eval_batch_size,
                num_workers=args.num_workers,
            )
            train_loaders.update({"online_eval": online_eval_loader})

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=2 * args.num_workers,
        )

    # check method
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    # build method
    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    if args.distiller:
        MethodClass = DISTILLERS[args.distiller](MethodClass)

    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # only one resume mode can be true
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1

    if args.resume_from_checkpoint:
        pass  # handled by the trainer
    elif args.pretrained_model:
        print(f"Loading previous task checkpoint {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # Apply torch.compile() for acceleration if requested and available
    if hasattr(args, 'compile_model') and args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model components with torch.compile() for acceleration...")
        
        compile_mode = getattr(args, 'compile_mode', 'default')
        print(f"Using compile mode: {compile_mode}")
        
        try:
            # Compile individual components instead of the entire Lightning module
            if hasattr(model, 'encoder'):
                model.encoder = torch.compile(model.encoder, mode=compile_mode)
            
            if hasattr(model, 'momentum_encoder'):
                model.momentum_encoder = torch.compile(model.momentum_encoder, mode=compile_mode)
                
            if hasattr(model, 'projector'):
                model.projector = torch.compile(model.projector, mode=compile_mode)
                
            if hasattr(model, 'head'):
                model.head = torch.compile(model.head, mode=compile_mode)
                
            if hasattr(model, 'momentum_head'):
                model.momentum_head = torch.compile(model.momentum_head, mode=compile_mode)
            
        except Exception as e:
            print(f"torch.compile() failed: {e}")
            
    elif hasattr(args, 'compile_model') and args.compile_model:
        print("torch.compile() requested but not available (PyTorch < 2.0)")

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{args.task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            reinit=True,
        )
        if args.task_idx == 0:
            wandb_logger.watch(model, log="gradients", log_freq=100)
            wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # Setup profiler if requested
    profiler = None
    if os.getenv("TORCH_PROFILER_ENABLED"):
        output_dir = os.getenv("TORCH_PROFILER_OUTPUT", "./torch_profiles")
        wait = int(os.getenv("TORCH_PROFILER_WAIT", 1))
        warmup = int(os.getenv("TORCH_PROFILER_WARMUP", 1))
        active = int(os.getenv("TORCH_PROFILER_ACTIVE", 3))
        repeat = int(os.getenv("TORCH_PROFILER_REPEAT", 2))
        
        profiler = PyTorchProfiler(
            dirpath=output_dir,
            filename="trace",
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        print(f"PyTorch profiler enabled: {output_dir}")

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        profiler=profiler,
        enable_checkpointing=False,
        detect_anomaly=True,
        gpus=args.gpus,  # Use our processed gpus list
    )

    model.current_task_idx = args.task_idx

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loaders, val_loader)


if __name__ == "__main__":
    main()
