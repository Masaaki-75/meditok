import os
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPImageProcessor
from ..meditok.meditok import get_meditok_args, MedITok



class MedITokEncoder(MedITok):

    def forward(self, image):
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_indices = self.quantizer.f_to_idx(img_tokens)
        img_tokens = self.quantizer.idx_to_f(img_indices)
        img_tokens = self.post_quant_proj(img_tokens)
        return img_tokens


class MedITokVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.args = get_meditok_args()
        self.image_processor = CLIPImageProcessor(
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            size={"shortest_edge": self.args.img_size},
            crop_size={"height": self.args.img_size, "width": self.args.img_size}
        )

        self.vision_tower = MedITokEncoder(self.args)
        model_weights = torch.load(self.vision_tower_name, map_location='cpu')
        self.vision_tower.load_state_dict(model_weights, strict=False)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.encoder.patch_embed.backbone.stem.conv1.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.encoder.patch_embed.backbone.stem.conv1.parameters()).device

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        #return self.vision_tower.embed_dim
        return self.vision_tower.encoder.embed_dim

    @property
    def num_patches_per_side(self):
        return self.args.img_size // self.args.patch_size

    @property
    def num_patches(self):
        return (self.args.img_size // self.args.patch_size) ** 2
