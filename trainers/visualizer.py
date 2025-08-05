import os
import time
import torch
import wandb
import warnings
import torchvision
import numpy as np
import seaborn as sns
from PIL import Image
from typing import Tuple
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

from utilities import misc
from utilities.dist import for_visualize
from trainers.logger import wandb_log


def pil_load(path: str, proposal_size):
    with open(path, 'rb') as f:
        img: Image.Image = Image.open(f)
        w: int = img.width
        h: int = img.height
        sh: int = min(h, w)
        if sh > proposal_size:
            ratio: float = proposal_size / sh
            w = round(ratio * w)
            h = round(ratio * h)
        img.draft('RGB', (w, h))
        img = img.convert('RGB')
    return img


class Visualizer(object):
    def __init__(self, enable: bool, device, trainer):
        self.enable = enable
        if enable:
            self.device = device
            self.trainer = trainer
            self.inp_B3HW: torch.Tensor = ...
            self.bound_mask: torch.Tensor = ...
            self.cmap_div: ListedColormap = sns.color_palette('mako', as_cmap=True)
            self.cmap_div: ListedColormap = sns.color_palette('icefire', as_cmap=True)
            self.cmap_seq = ListedColormap(sns.color_palette('ch:start=.2, rot=-.3', as_cmap=True).colors[::-1])
            self.cmap_seq: ListedColormap = sns.color_palette('RdBu_r', as_cmap=True)
            self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)

    @for_visualize
    def vis_prologue(self, inp_B3HW: torch.Tensor) -> None:
        if not self.enable:
            return
        self.inp_B3HW = inp_B3HW

    def denormalize(self, BCHW):
        # BCHW = BCHW * self.data_s
        # BCHW += self.data_m
        return BCHW.add(1).mul_(0.5).clamp_(0, 1)

    @for_visualize
    def vis(self, epoch, png_path='', report_wandb=False) -> Tuple[float, float]:
        if not self.enable:
            return -1., -1.
        vis_stt = time.time()
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # get recon
        B = self.inp_B3HW.shape[0]
        with torch.inference_mode():
            assert hasattr(self.trainer, 'model'), f"Trainer class has no self.model implemented! Check its definition."
            self.trainer.model.eval()
            rec_B3HW = misc.unwrap_model(self.trainer.model).img_to_reconstructed_img(self.inp_B3HW)
            self.trainer.model.train()

            L1 = F.l1_loss(rec_B3HW, self.inp_B3HW).item()
            Lpip = self.trainer.lpips_loss(rec_B3HW, self.inp_B3HW).item()
            diff = (L1 + Lpip) / 2

        # viz
        H, W = rec_B3HW.shape[-2], rec_B3HW.shape[-1]
        cmp_grid = torchvision.utils.make_grid(
            self.denormalize(torch.cat((self.inp_B3HW, rec_B3HW), dim=0)), padding=0, pad_value=1, nrow=B)

        if report_wandb:
            wandb_log({'Vis_Lnll': diff})
            wandb_log({'Vis_img': wandb.Image(cmp_grid)})

        if png_path:
            chw = cmp_grid.permute(1, 2, 0).mul_(255).cpu().numpy()
            chw = PImage.fromarray(chw.astype(np.uint8))
            if not chw.mode == 'RGB':
                chw = chw.convert('RGB')
            chw.save(png_path)

        print(f'[vis] {L1=:.3f}, {Lpip=:.3f}, cost={time.time() - vis_stt:.2f}s', force=True)
        warnings.resetwarnings()
        return


def get_boundary(patch_size, needs_loss, boundary_wid=3):  # vis_img: BCHW, needs_loss: BL
    """
    get the boundary of `False`-value connected components on given boolmap `needs_loss`
    """
    B, L = needs_loss.shape
    hw = round(L ** 0.5)
    boolmap = (~needs_loss).view(B, 1, hw, hw)  # BL => B1hw
    boolmap = boolmap.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)  # B1hw => B1HW

    k_size = boundary_wid * 2 + 1
    conv_kernel = torch.ones(1, 1, k_size, k_size).to(boolmap.device)
    bound_mask = F.conv2d(boolmap.float(), conv_kernel, padding=boundary_wid)
    bound_mask = ((bound_mask - k_size ** 2).abs() < 0.1) ^ boolmap  # B1HW

    return bound_mask.float()


def setup_visualizer(args, trainer, preprocess_val):
    vis_imgs = []
    for img in os.listdir(args.vis_img_dir):
        img = os.path.join(args.vis_img_dir, img)
        img = pil_load(img, args.img_size * 2)
        img = preprocess_val(img)
        if isinstance(img, Image.Image):
            img = np.array(img, dtype=np.float32) / 255.0 * 2 - 1
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
        vis_imgs.append(img)
    vis_imgs = torch.stack(vis_imgs, dim=0).to(args.device, non_blocking=True)
    visualizer = Visualizer(True, args.device, trainer)
    visualizer.vis_prologue(vis_imgs)
    return visualizer
