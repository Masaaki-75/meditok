import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPImageProcessor

import sys

sys.path.append("../../../../..")
from models.meditok import get_meditok_args
from models.vitamin import GeGluMlp
from models.quant import VectorQuantizerM
from models.vqvae import AttnProjection


class MedITokEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            reg_tokens=args.num_query,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=False)

        if args.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, args.num_codebooks)
        else:
            raise NotImplementedError

        self.quantizer = VectorQuantizerM(
            vocab_size=args.vocab_size,
            vocab_width=args.vocab_width,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )

        if args.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, args.num_codebooks)
        else:
            raise NotImplementedError

        self.embed_dim = self.encoder.embed_dim

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
        return self.vision_tower.embed_dim

    @property
    def num_patches_per_side(self):
        return self.args.img_size // self.args.patch_size

    @property
    def num_patches(self):
        return (self.args.img_size // self.args.patch_size) ** 2
