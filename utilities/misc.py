import datetime
import functools
import glob
import os
import subprocess
import sys
import pytz
import torch
import threading
import time
from typing import List, Tuple

from utilities import dist

os_system = functools.partial(subprocess.call, shell=True)


def echo(info):
    os_system(
        f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')


def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class DistLogger(object):
    def __init__(self, lg):
        self._lg = lg

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass

    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._lg is not None else DistLogger.do_nothing


class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try:
            import tensorflow_io as tfio
        except:
            pass
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def loggable(self):
        return self.step == 0 or (self.step + 1) % 500 == 0

    def update(self, head='scalar', step=None, **kwargs):
        if step is None:
            step = self.step
            if not self.loggable(): return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            self.writer.add_scalar(f'{head}/{k}', v, step)

    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        try:
            self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
        except Exception as e:
            print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')

    def log_image(self, tag, img_chw, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        self.writer.add_image(tag, img_chw, step, dataformats='CHW')

    def flush(self):
        self.writer.flush()

    def close(self):
        print(f'[{type(self).__name__}] file @ {self.log_dir} closed')
        self.writer.close()


class TouchingDaemon(threading.Thread):
    def __init__(self, files: List[str], sleep_secs: int, verbose=False):
        super().__init__(daemon=True)
        self.files = tuple(files)
        self.sleep_secs = sleep_secs
        self.is_finished = False
        self.verbose = verbose

        f_back = sys._getframe().f_back
        file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
        self.print_prefix = f' ({file_desc}, line{f_back.f_lineno:-4d}) @daemon@ '

    def finishing(self):
        self.is_finished = True

    def run(self) -> None:
        # stt, logged = time.time(), False
        kw = {}
        if dist.initialized(): kw['clean'] = True

        stt = time.time()
        if self.verbose: print(
            f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] start touching {self.files} per {self.sleep_secs}s ...',
            **kw)
        while not self.is_finished:
            for f in self.files:
                if os.path.exists(f):
                    try:
                        os.utime(f)
                        fp = open(f, 'a')
                        fp.close()
                    except:
                        pass
                    # else:
                    #     if not logged and self.verbose and time.time() - stt > 180:
                    #         logged = True
                    #         print(f'[TouchingDaemon tid={threading.get_native_id()}] [still alive ...]')
            time.sleep(self.sleep_secs)

        if self.verbose: print(
            f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] finish touching after {time.time() - stt:.1f} secs {self.files} per {self.sleep_secs}s. ',
            **kw)


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)


def auto_resume(args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, dict, dict]:
    info = []
    file = os.path.join(args.local_out_dir_path, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {file}')
        info.append(f'[auto_resume quit]')
        return info, 0, 0, {}, {}
    else:
        info.append(f'[auto_resume] load ckpt from @ {all_ckpt[0]} ...')
        ckpt = torch.load(all_ckpt[0], map_location='cpu')
        ep, it = ckpt['epoch'], ckpt['iter']
        info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
        return info, ep, it, ckpt['trainer'], ckpt['args']


def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image

    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path
