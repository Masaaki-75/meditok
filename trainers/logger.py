import os
import time
import wandb
import torch
import GPUtil
import colorama
import datetime
import numpy as np
from math import log10
from collections import deque
from typing import Dict, List
import torch.distributed as tdist
from collections import defaultdict
from typing import Iterator, List, Tuple

from utilities import config, misc, dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0

    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)

    @property
    def global_avg(self):
        return self.total / (self.count or 1)

    @property
    def max(self):
        return max(self.deque) if len(self.deque) else 0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0

    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        time_str1 = str(datetime.timedelta(seconds=round(remain_secs)))
        time_str2 = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
        return remain_secs, time_str1, time_str2

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self, cur_epoch, total_epoch, delimiter="\t"):
        self.cur_epoch = cur_epoch
        self.total_epoch = total_epoch
        self.delimiter = delimiter
        self.meters = defaultdict(SmoothedValue)
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(v, 'item'):
                v = v.item()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def monitor_enumerate(self, dataloader, start_iter, num_iters, print_freq=100):
        start_time = time.time()
        time_stamp = time.time()

        log_msg = ['[Ep]: [{cur_epoch}/{total_epoch}]', '[{cur_iter}/{num_iters}]']
        log_msg += ['{meters}', 'eta: {eta}', 'iter_time: {iter_time}', 'data_time: {data_time}']
        log_msg = self.delimiter.join(log_msg)

        if isinstance(dataloader, Iterator):
            for cur_iter in range(start_iter, num_iters):
                sample = next(dataloader)
                self.data_time.update(time.time() - time_stamp)
                yield cur_iter, sample
                self.iter_time.update(time.time() - time_stamp)
                if cur_iter % print_freq == 0:
                    eta_iters = (self.total_epoch - self.cur_epoch) * num_iters - cur_iter
                    eta_seconds = self.iter_time.global_avg * eta_iters
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        cur_epoch=self.cur_epoch,
                        total_epoch=self.total_epoch,
                        cur_iter=cur_iter,
                        num_iters=num_iters,
                        meters=str(self),
                        eta=eta_string,
                        iter_time=str(self.iter_time),
                        data_time=str(self.data_time)
                    ), flush=True)
                time_stamp = time.time()
        else:
            for cur_iter, sample in enumerate(dataloader):
                if cur_iter < start_iter:
                    time_stamp = time.time()
                    continue
                self.data_time.update(time.time() - time_stamp)
                yield cur_iter, sample
                self.iter_time.update(time.time() - time_stamp)
                if cur_iter % print_freq == 0:
                    eta_iters = (self.total_epoch - self.cur_epoch) * num_iters - cur_iter
                    eta_seconds = self.iter_time.global_avg * eta_iters
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        cur_epoch=self.cur_epoch,
                        total_epoch=self.total_epoch,
                        cur_iter=cur_iter,
                        num_iters=num_iters,
                        meters=str(self),
                        eta=eta_string,
                        iter_time=str(self.iter_time),
                        data_time=str(self.data_time)
                    ), flush=True)
                time_stamp = time.time()

        epoch_time = time.time() - start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print(
            f'[Ep]: [{self.cur_epoch}/{self.total_epoch}]  Total cost: {epoch_time_str} ({epoch_time / (num_iters - start_iter):.3f} s / it)',
            flush=True)


class ProfileLogger(object):
    def __init__(self, args, log_freq):
        self.bs = args.global_bs
        self.log_freq = log_freq
        self.max_nvidia_smi = 0
        self.speed_ls = deque(maxlen=128)
        self.last_t_perf = time.perf_counter()

    def log(self, cur_iter):
        if (cur_iter + 1) % self.log_freq == 0:
            self.speed_ls.append((time.perf_counter() - self.last_t_perf) / self.log_freq)
            iter_speed = float(np.median(self.speed_ls))
            img_per_sec = self.bs / iter_speed
            img_per_day = img_per_sec * 3600 * 24 / 1e6
            self.max_nvidia_smi = max(self.max_nvidia_smi, max(gpu.memoryUsed for gpu in GPUtil.getGPUs()) / 1024)
            mem_infos_dict = torch.cuda.memory_stats()
            memory_allocated = round(mem_infos_dict['allocated_bytes.all.current'] / 1024 ** 3, 2)
            memory_reserved = round(mem_infos_dict['reserved_bytes.all.current'] / 1024 ** 3, 2)
            max_memory_allocated = round(mem_infos_dict['allocated_bytes.all.peak'] / 1024 ** 3, 2)
            max_memory_reserved = round(mem_infos_dict['reserved_bytes.all.peak'] / 1024 ** 3, 2)
            num_alloc_retries = mem_infos_dict['num_alloc_retries']

            tails = list(self.speed_ls)[-10:]
            print(
                colorama.Fore.LIGHTCYAN_EX +
                f"[profiling]  "
                f"speed: {iter_speed:.3f} ({min(tails):.3f}~{max(tails):.2f}) sec/iter  |  "
                f"{img_per_sec:.1f} imgs/sec  |  "
                f"{img_per_day:.2f}M imgs/day  ||  "
                # f"{img_per_day * (args.img_size // trainer.vae_wo_ddp.downsample_ratio)**2 / 1e3:.2f}B token/day  ||  "
                f"Peak nvidia-smi: {self.max_nvidia_smi:.2f} GB  ||  "
                f"PyTorch mem - "
                f"alloc: {memory_allocated:.2f}  |  "
                f"max_alloc: {max_memory_allocated:.2f}  |  "
                f"reserved: {memory_reserved:.2f}  |  "
                f"max_reserved: {max_memory_reserved:.2f}  |  "
                f"num_alloc_retries: {num_alloc_retries}" +
                colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL,
                flush=True
            )
            self.last_t_perf = time.perf_counter()


def wandb_log(data, step=None, log_freq=None, commit=None, sync=None):
    if not dist.is_master():
        return
    if step is not None and log_freq is not None:
        if step % log_freq == 0:
            wandb.log(data, step=step, commit=commit, sync=sync)
        else:
            return
    else:
        wandb.log(data, step=step, commit=commit, sync=sync)


def create_tb_log(args: config.Args):
    tb_log: misc.TensorboardLogger
    with_tb_log = dist.is_master()
    if with_tb_log:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_log = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_online,
                filename_suffix=f'_{args.trial_id}_{misc.time_str("%m%d_%H%M")}'
            )
        )
        tb_log.flush()
    else:
        # noinspection PyTypeChecker
        tb_log = misc.DistLogger(None)
    dist.barrier()
    return tb_log


def get_param_for_log(model_name_3letters: str, named_parameters) -> Dict[str, List[float]]:
    dists = defaultdict(list)

    for n, p in named_parameters:
        n: str
        if p.grad is None: continue
        post = 'B' if ('.bias' in n or '_bias' in n) else 'W'

        if 'gpt' in model_name_3letters:
            if 'word' in n:
                tag = '0-word'
            elif 'norm0_ve' in n:
                tag = '0-norm0_ve'
            elif 'norm0_cond' in n:
                tag = '0-norm0_cond'
            elif 'start' in n:
                tag, post = '1-start', 'T'
            elif 'class_emb' in n:
                tag, post = '1-cls_emb', 'W'
            elif 'cls_token' in n:
                tag, post = '1-cls', 'T'
            elif 'cfg_uncond' in n:
                tag, post = '1-cond_cfg', 'T'
            elif 'cond_sos' in n:
                tag, post = '1-cond_sos', 'W'
            elif 'text_proj_for_sos' in n:
                tag = '1-text_sos'
            elif 'text_proj_for_ca' in n:
                tag = '1-text_ca'

            elif 'ca_rpb' in n:
                tag, post = '2-ca_rpb', 'T'
            elif 'sa_rpb' in n:
                tag, post = '2-sa_rpb', 'T'
            elif 'start_p' in n or 'pos_start' in n:
                tag, post = '2-pos_st', 'T'
            elif 'abs_pos_embed' in n:
                tag, post = '2-pos_abs', 'T'
            elif 'pos_mlp' in n:
                tag = '2-pos_mlp'
            elif 'lvl_embed' in n:
                tag, post = '2-pos_lvl', 'T'
            elif 'pos_1LC' in n:
                tag, post = '2-pos_1LC', 'T'
            elif 'pos_task' in n:
                tag, post = '2-pos_task', 'T'

            elif 'get_affine_4num' in n:
                tag = '1-freq_aff'
            elif 'freq_proj' in n:
                tag, post = '1-freq_prj', 'W'
            elif 'task_token' in n:
                tag, post = '1-task', 'T'
            elif 'adaIN_elin' in n:
                tag = '4-aIN_elin'
            elif 'shared_ada_lin' in n:
                tag = '2-shared_ada_lin'
            elif 'ada_lin' in n:
                tag = '4-ada_lin'
            elif 'ada_gss' in n:
                tag, post = '4-ada_gss', 'T'
            elif 'ada_gamma' in n:
                tag, post = '4-aIN_elin', 'GA'
            elif 'ada_beta' in n:
                tag, post = '4-aIN_elin', 'BE'
            elif 'moe_bias' in n:
                tag, post = '4-moe_bias', 'B'

            elif 'scale_mul' in n:
                tag, post = '3-2-scale', 'LogMul'
            elif 'norm1' in n:
                tag = '3-1-norm1'
            elif 'sa.' in n or 'attn.' in n:
                tag = '3-2-sa'
            elif 'ca.' in n:
                tag = '3-2-ca'
            elif 'gamma1' in n:
                tag, post = '3-3-gam1', 'GA'
            elif 'ca_norm' in n:
                tag = '3-2-ca_norm'
            elif 'ca_gamma' in n:
                tag, post = '3-3-ca_gam', 'GA'

            elif 'norm2' in n:
                tag = '4-1-norm1'
            elif 'ffn.' in n:
                tag = '4-2-ffn'
            elif 'gamma2_last' in n:
                tag, post = '4-3-gam2-last', 'GA'
            elif 'gamma2' in n:
                tag, post = '4-3-gam2', 'GA'

            elif 'head_nm' in n:
                tag = '5-headnm'
            elif 'head0' in n:
                tag = '5-head0'
            elif 'head_bias' in n:
                tag = '5-head_b', 'B'
            elif 'head' in n:
                tag = '5-head'
            elif 'up' in n:
                tag = '5-up'

            else:
                tag = f'___{n}___'

        elif 'vae' in model_name_3letters:
            if 'encoder.' in n or 'decoder.' in n:
                i, j = (0, 'enc') if 'encoder.' in n else (7, 'dec')
                if 'conv_in' in n:
                    tag = f'{0 + i}-{j}_cin'
                elif 'down.' in n and '.block' in n:
                    tag = f'{1 + i}-{j}_res'
                elif 'down.' in n and '.downsample' in n:
                    tag = f'{1 + i}-{j}_cdown'
                elif 'down.' in n and '.attn' in n:
                    tag = f'{1 + i}-{j}_attn'
                elif 'up.' in n and '.block' in n:
                    tag = f'{1 + i}-{j}_res'
                elif 'up.' in n and '.upsample' in n:
                    tag = f'{1 + i}-{j}_cup'
                elif 'up.' in n and '.attn' in n:
                    tag = f'{1 + i}-{j}_attn'
                elif 'mid.' in n and '.block' in n:
                    tag = f'{2 + i}-{j}_mid_res'
                elif 'mid.' in n and '.attn' in n:
                    tag = f'{2 + i}-{j}_mid_at'
                elif 'norm_out' in n:
                    tag = f'{3 + i}-{j}_nout'
                elif 'conv_out' in n:
                    tag = f'{3 + i}-{j}_cout'
                else:
                    tag = f'3-enc___{n}___'
            elif 'quant_conv' in n:
                tag = f'4-quan_pre'
            elif 'post_quant_conv' in n:
                tag = f'6-quan_post'
            elif 'quant_proj' in n:
                tag = f'5-0-quan_pre_proj'
            elif 'quant_resi' in n:
                tag = f'5-2-quan_post_resi'
            elif 'post_quant_proj' in n:
                tag = f'5-2-quan_post_proj'
            elif 'quant' in n and 'norm_scale' in n:
                tag = f'5-1-quan_norm_scale'
            elif 'quant' in n and 'embed' in n:
                tag = f'5-1-quan_emb'
            else:
                tag = f'uk___{n}___'

        elif 'disc' in model_name_3letters or 'dsc' in model_name_3letters:  # discriminator
            if 'dwt' in n:
                tag = '0-dwt'
            elif 'from' in n:
                tag = '0-from'
            elif 'resi' in n:
                tag = '0-resi'
            elif 'fpn' in n:
                tag = '1-fpn'
            elif 'down' in n:
                tag = '2-down'
            elif 'head_conv' in n:
                tag = '3-head_conv'
            elif 'head_cls' in n:
                tag = '4-head_cls'
            elif 'norm.' in n:
                tag = 'x_norm'
            elif 'head.' in n:  # DinoDisc
                tag = n.split('heads.')[-1][0]
                if p.ndim == 3:
                    tag += '.conv1d'
                else:
                    tag += '.other'
            else:  # StyleGanDisc
                tag = n.rsplit('.', maxsplit=1)[0]
                if p.ndim == 4:
                    tag += '.conv'
                else:
                    tag += '.other'

        else:
            tag = f'uk___{n}___'

        m = p.grad.norm().item()
        m = log10(m) if m > 1e-9 else -10
        dists[f'Gnorm_{model_name_3letters}.{tag}.{post}'].append(m)
        m = p.data.abs().mean().item()
        m = log10(m) if m > 1e-9 else -10
        dists[f'Para_{model_name_3letters}.{tag}.{post}'].append(m)

    return dists
