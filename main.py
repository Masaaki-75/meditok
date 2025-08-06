import os
import gc
import sys
import time
import glob
import torch
import wandb
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP

from utilities import config, dist
from utilities.lpips import LPIPS
from utilities.loss import ClipLoss
from datasets import load_data
from datasets.transforms import build_image_transform
from models.discrim import build_discriminator
from models.wrapper import build_meditok_wrapper
from trainers.scheduler import LRScheduler
from trainers.optimizer import build_optimizer
from trainers.visualizer import setup_visualizer
from trainers.trainer import Trainer, train_one_ep
from local_openclip.tokenizer import get_biomedclip_tokenizer_offline, tokenize



def maybe_auto_resume(args: config.Args, pattern='ckpt*.pth'):
    if len(args.resume_from):
        resume = args.resume_from
        print(f'[auto_resume] Load from args.resume @ {resume} ...')
    else:
        all_ckpt = glob.glob(os.path.join(args.output_dir, pattern), recursive=False)
        all_ckpt = sorted(all_ckpt, key=os.path.getmtime, reverse=True)
        if len(all_ckpt) == 0:
            resume = None
            print(f'[auto_resume] NO ckpt found @ {pattern}')
            print(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            print(f'[auto_resume] Auto resume from @ {resume} ...')

    if resume is not None:
        print(f'[auto_resume] Load networks only (w/o optimizers)? @ {args.resume_net_only} ...')
        try:
            ckpt_ = torch.load(resume, map_location='cpu')
            ckpt = {}
            dist.barrier()

            is_bare_weights = 'trainer' not in ckpt_
            if is_bare_weights:
                # loading bare model weights
                print(f'[auto_resume] Load BARE weights @ {resume}')
                ckpt['epoch'] = 0
                ckpt['iter'] = 0
                ckpt['args'] = args
                ckpt['trainer'] = {'model': ckpt_}
            else:
                # loading the whole checkpoint
                ckpt = ckpt_
                if args.resume_net_only:
                    ckpt['epoch'] = 0
                    ckpt['iter'] = 0
                    ckpt['args'] = args

            resume_epoch = ckpt['epoch']
            resume_iter = ckpt['iter']

            if resume_epoch == args.epoch:
                print(f'[auto_resume] Training finished, skipping ...\n\n')
                exit()
            else:
                print(f'[auto_resume success] Resume ep{resume_epoch} & it{resume_iter} @ {resume}')
                return ckpt
        except Exception as e:
            print(f'[auto_resume] Failed, {e} @ {resume}')
            return {}
    else:
        return {}



def main():
    args = config.init_dist_and_get_args()
    print(f'[args] initial args:\n{str(args)}')

    # resume ckpt
    ckpt = maybe_auto_resume(args, 'ckpt*.pth')
    start_iter = ckpt.get('iter', 0)
    start_epoch = ckpt.get('epoch', 0)
    trainer_state = ckpt.get('trainer', {})

    # load data
    print(f'[data] Load data...\n') 

    if args.use_biomedclip:
        tokenizer = get_biomedclip_tokenizer_offline()
    else:
        tokenizer = partial(tokenize, context_length=args.text_context_length)
    data = load_data(args, epoch=start_epoch, iters=start_iter, tokenizer=tokenizer)

    # build models
    model = build_meditok_wrapper(args)
    disc = build_discriminator(args)

    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
            freeze_logit_scale=args.freeze_logit_scale,
        )
    
    if args.lock_visual_proj:
        model.lock_visual_projector()

    print(f'[model] Model #params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} (M)')
    print(f'[model] Disc #params {sum(p.numel() for p in disc.parameters()) / 1e6:.2f} (M)')

    # build optimizers & scheduler
    model_optim = build_optimizer(args, 'model', model)
    disc_optim = build_optimizer(args, 'dis', disc)

    max_iter = args.epoch * data['train'].num_batches
    warmup_iter = args.warmup_ep * data['train'].num_batches
    disc_max_iter = max_iter - args.disc_start_ep * data['train'].num_batches
    disc_warmup_iter = args.disc_warmup_ep * data['train'].num_batches

    model_schedule = {
        'lr': args.lr,
        'type': args.schedule,
        'start_factor': args.lr_start_ratio,
        'end_factor': args.lr_end_ratio,
        'warmup_iter': warmup_iter,
        'max_iter': max_iter,
    }
    disc_schedule = {
        'lr': args.disc_lr,
        'type': args.schedule,
        'start_factor': args.lr_start_ratio,
        'end_factor': args.disc_lr_end_ratio,
        'warmup_iter': disc_warmup_iter,
        'max_iter': disc_max_iter,
    }
    model_scheduler = LRScheduler(model_optim.optimizer, model_schedule)
    disc_scheduler = LRScheduler(disc_optim.optimizer, disc_schedule)

    # build loss
    clip_loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        use_horovod=False,
    )
    lpips_loss = LPIPS(args.lpips_path).to(args.device)

    # torch compile model
    if args.compile_model:
        model = torch.compile(model, backend='inductor')
        disc = torch.compile(disc, backend='inductor')
        lpips_loss = torch.compile(lpips_loss, backend='inductor')

    # distributed wrapper
    model = DDP(model, device_ids=[dist.get_local_rank()], static_graph=args.ddp_static)
    disc = DDP(disc, device_ids=[dist.get_local_rank()], static_graph=args.ddp_static)

    # build trainer
    trainer = Trainer(
        args=args,
        model=model,
        disc=disc,
        model_optim=model_optim,
        disc_optim=disc_optim,
        clip_loss=clip_loss,
        lpips_loss=lpips_loss,
    )
    
    if trainer_state:
        trainer.load_state_dict(
            trainer_state, 
            strict=False, 
            resume_net_only=args.resume_net_only,
            ignore_text_params=args.ignore_text_params,
            core_weights_only=args.core_weights_only,
        )

    # setup visualizer
    vis_transform = build_image_transform(args.img_size, is_train=False)
    visualizer = setup_visualizer(args, trainer, vis_transform)

    # setup wandb
    if args.report_wandb and dist.is_master():
        wandb.init(
            project='meditok',
            resume='auto',
            save_code=True,
            id=args.run_id,
            name=args.exp_name,
            notes=args.wandb_notes,
            config=args.state_dict()
        )

    # train
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    print(f'[train] Exp output directory: {args.output_dir}')
    print(f'[train] Start exp at epoch {start_epoch} iter {start_iter}')

    for epoch in range(start_epoch, args.epoch):
        gc.collect()
        print(f'[dataloader] set_epoch({epoch})')
        data['train'].set_epoch(epoch)

        start_iter = start_iter if epoch == start_epoch else 0
        print(f'[train] Start training ({epoch})')
        stats = train_one_ep(
            args=args,
            data=data,
            epoch=epoch,
            trainer=trainer,
            start_iter=start_iter,
            model_scheduler=model_scheduler,
            disc_scheduler=disc_scheduler,
            visualizer=visualizer
        )

    if dist.is_master():
        ckpt_path = os.path.join(args.output_dir, 'ckpt-last.pth')
        torch.save({
            'args': args.state_dict(),
            'epoch': args.epoch, 'iter': 0,
            'trainer': trainer.state_dict(),
        }, ckpt_path)
    dist.barrier()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print(f"[train] Total Training Time: {total_time},\t Lg: {stats['Lnll']:.3f},\t Ld: {stats['Ld']:.3f}")

    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()


if __name__ == '__main__':
    main()
