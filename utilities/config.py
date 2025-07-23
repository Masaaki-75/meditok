import os
import sys
import torch
import random
import numpy as np
from tap import Tap
from typing import Optional, Union
from collections import OrderedDict

from utilities import dist


class Args(Tap):
    model: str = 'vitamin_large' # 'vitamin_base', 'vitamin_large', xxx
    exp_name: str = 'unitok_large'
    output_dir: str = 'local_output'
    resume_from: str = ''  # if specified, load this checkpoint; if not, load the latest checkpoint in output_dir (if exists)
    resume_net_only: bool = False  # [NOTE] I modified here
    lpips_path: str = 'external/lpips_with_vgg.pth'
    dino_path: str = 'external/dinov2_vits14_pretrain.pth'
    fid_eval_src: str = ''
    fid_eval_dst: str = ''
    vis_img_dir: str = 'asset/vis_imgs/'
    fid_feature_extractor: str = 'external/weights-inception-2015-12-05-6726825d.pth'
    clip_pretrain_path: str = ''

    # speed-up
    fp16: bool = False  # whether to use FP16
    bf16: bool = True  # whether to use BF16
    tf32: bool = True  # whether to use TensorFloat32
    compile_model: bool = False  # whether to use torch.compile()
    ddp_static: bool = False  # whether to use static graph in DDP
    grad_ckpt: bool = True  # gradient checkpointing
    grad_accu: int = 1  # gradient accumulation
    device: str = 'cpu' # will be set automatically
    dtype: torch.dtype = torch.float32 # will be set automatically

    # data
    train_data: str = None
    val_data: str = None
    train_root: str = None
    val_root: str = None
    dataset_type: str = 'csvimg'
    csv_img_key: str = 'image_id'
    csv_caption_key: str = 'caption'
    csv_separator: str = ','
    subset_ratio: float = 1.0
    img_size: int = 256
    resize_ratio: float = 1.125  # only applicable to 'img' dataset_type
    hflip: bool = False
    workers: int = 8  # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    train_num_samples: int = None
    train_data_upsampling_factors: str = None
    val_num_samples: int = None
    dataset_resampled: bool = False
    use_aug: bool = False
    prob_flip: float = 0.5  # probability for data augmentation: image flipping
    prob_rot: float = 0.5  # probability for data augmentation: image rotation
    prob_grayscale: float = 0.1  # probability for data augmentation: image rotation

    # quantizer
    vocab_size: int = 32768
    vocab_width: int = 64
    vocab_norm: bool = True
    vq_beta: float = 0.25  # commitment loss weight
    num_codebooks: int = 8
    quant_proj: str = 'attn'

    # model
    embed_dim: int = 768
    num_query: int = 0
    use_clip_pretrain: bool = False
    patch_size: int = 16
    drop_path: float = 0.1
    text_width: int = 768
    text_heads: int = 12
    text_layers: int = 12
    text_vocab_size: int = 49408
    text_context_length: int = 77

    # CLIP
    local_loss: bool = True
    gather_with_grad: bool = True
    pretrained_clip: str = None
    pretrained_clip_text: str = None
    lock_text: bool = False
    lock_text_unlocked_layers: int = 0
    lock_text_freeze_layer_norm: bool = False
    force_custom_text: bool = False
    force_custom_vision: bool = False
    zeroshot_eval_freq: int = 1

    # discriminator
    dino_depth: int = 12
    dino_kernel_size: int = 9
    disc_norm: str = 'gn'  # gn: group norm, bn: batch norm, sbn: sync batch norm, hbn: hybrid sync batch norm
    disc_aug_prob: float = 1.0
    disc_specnorm: bool = False
    step_disc_every: int = 1
    
    # initialization
    vae_init: float = -0.5  # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
    vocab_init: float = -1  # <0: uniform(-abs(init)*base, abs(init)*base), where base = 20/vocab_size; >0: trunc_normal_(std=init)
    disc_init: float = -0.5  # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)

    # optimization
    epoch: int = 1  # number of epochs
    local_bs: int = 64  # batch size per device; if this is specified, --global_bs will be ignored
    global_bs: int = 0  # global batch size (exclusive to --local_bs)
    lr: float = 5e-4  # learning rate
    wd: float = 0.02  # weight decay
    disc_lr: float = 2e-5  # disc lr
    disc_wd: float = 0.2
    grad_clip: float = 10  # <=0 for not using grad clip
    ema: float = 0.9999  # ema ratio
    warmup_iter: int = None
    warmup_ep: float = 0.01  # lr warmup: epochs
    disc_start_ep: float = 0.375  # start using disc loss for VAE after xxx epochs;
    disc_warmup_ep: float = 0.03  # disc loss warm up epochs;
    schedule: str = 'cos'  # lr schedule type
    lr_start_ratio: float = 0.  # lr warmup: initial lr ratio
    lr_end_ratio: float = 0.1  # lr schedule: final lr ratio
    disc_lr_end_ratio: float = 0.1
    custom_lr_multiplier: float = None
    optimizer: str = 'adamw'
    optim_eps: float = 1e-6
    fuse_opt: bool = False  # whether to use fused optimizer
    optim_beta: str = '0.9_0.95'  # beta1, beta2 of optimizer
    disc_optim_beta: str = '0.5_0.9'  # beta1, beta2 of disc optimizer

    # loss
    l1: float = 0.2  # L1 rec loss weight
    l2: float = 1.0  # L2 rec loss weight
    lp: float = 1.0  # lpips loss weight
    lpr: int = 48    # only calculate lpips >= this image resolution
    ld: float = 0.4  # discriminator loss weight; if <0: NO ADAPTIVE WEIGHT
    le: float = 0.0  # VQ entropy loss weight
    lq: float = 1.0
    lc: float = 1.0  # CLIP loss weight
    lcos: float = 0.0  # cosine loss weight
    e_temp: float = 0.01
    gada: int = 1
    bcr: float = 4.  # balanced Consistency Regularization, used on small dataset with low reso, StyleSwin: 10.0
    bcr_cut: float = 0.2  # cutout ratio (0.5: 50% width)
    dcrit: str = 'hg'  # hg hinge, sp softplus, ln linear

    # wandb log
    report_wandb: bool = True
    wandb_notes: str = None
    run_id: str = None

    # debug
    eval_per_epoch: int = 4
    dbg_unused_param: bool = False
    dbg_nan: bool = False  # 'KEVIN_LOCAL' in os.environ
    seed: int = None
    deterministic: bool = False
    same_seed_for_all_ranks: int = 0  # this is only for distributed sampler

    # my
    use_biomedclip: bool = False
    vision_as_text: bool = False
    ignore_text_params: bool = False
    ct_bias: int = 1024
    freeze_logit_scale: bool = False
    pretrained_core_path: str = None

    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist.get_rank() * 10000
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:  # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:  # these are not serializable
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            try:
                setattr(self, k, v)
            except Exception as e:
                print(f'k={k}, v={v}')
                raise e

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:  # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break

    args = Args(explicit_bool=True).parse_args(known_only=True)
    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')

    # init torch distributed
    os.makedirs(args.output_dir, exist_ok=True)
    dist.init_distributed_mode(local_out_path=args.output_dir, timeout_minutes=30)

    # set env
    args.set_tf32(args.tf32)
    args.seed_everything()
    args.device = dist.get_device()

    # update args
    if args.local_bs == 0:
        args.local_bs = max(1, round(args.global_bs / args.grad_accu / dist.get_world_size()))
    args.global_bs = args.local_bs * dist.get_world_size()
    if args.fp16 or args.bf16:
        args.dtype = torch.float16 if args.fp16 else torch.bfloat16

    return args
