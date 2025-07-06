import os
import sys
sys.path.append('..')
import timm
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.vqvae import AttnProjection
from models.quant import VectorQuantizerM
from models.vitamin import GeGluMlp, ViTaminDecoder


class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)


class MedITok(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_query = args.num_query

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
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, self.encoder.embed_dim // args.vocab_width)
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
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, self.encoder.embed_dim // args.vocab_width)
        else:
            raise NotImplementedError

        self.decoder = ViTaminDecoder(
            args.model,
            num_query=args.num_query,
            img_size=args.img_size,
            drop_path=args.drop_path,
            grad_ckpt=args.grad_ckpt,
        )
        
        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        self.projection = nn.Linear(self.encoder.embed_dim, args.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.encoder.set_grad_checkpointing(args.grad_ckpt)

    def forward(self, img, vae_bs, text=None, ret_usages=False):
        img_tokens = self.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            img_tokens = self.quant_proj(img_tokens)
            img_tokens, vq_loss, _, usages = self.quantizer(img_tokens)
            img_tokens = self.post_quant_proj(img_tokens)
        img_rec = self.decoder(img_tokens[:vae_bs]).float()

        feats = img_tokens.mean(dim=1)
        feats = self.projection(self.fc_norm(feats))
        feats = F.normalize(feats, dim=-1)

        output_dict = {
            "img_rec": img_rec,
            "vq_loss": vq_loss,
            "codebook_usages": usages,
            "features": feats,
            "logit_scale": self.logit_scale.exp()
        }
        return output_dict
    
    def encode_image_vq(self, image):
        img_tokens = self.encoder(image)  # [batch_size, 256, 1024] for 3x256x256 inputs
        img_tokens = self.quant_proj(img_tokens)  # [batch_size, 256, 64]
        img_indices = self.quantizer.f_to_idx(img_tokens)  # [batch_size, 8, 256]
        img_tokens = self.quantizer.idx_to_f(img_indices)  # [batch_size, 256, 64]
        return img_tokens

    def encode_image(self, image, normalize=False):
        img_tokens = self.encoder(image)  # [batch_size, 256, 1024] for 3x256x256 inputs
        img_tokens = self.quant_proj(img_tokens)  # [batch_size, 256, 64]
        img_indices = self.quantizer.f_to_idx(img_tokens)  # [batch_size, 8, 256]  # 16*16*8, 一个patch编码成8个code
        img_tokens = self.quantizer.idx_to_f(img_indices)  # [batch_size, 256, 64]
        img_tokens = self.post_quant_proj(img_tokens)  # [batch_size, 256, 1024]

        features = img_tokens.mean(dim=1)
        features = self.projection(self.fc_norm(features))  # [batch_size, 768]
        return F.normalize(features, dim=-1) if normalize else features
    
    def img_to_idx(self, img):
        features = self.encoder(img).float()
        features = self.quant_proj(features)
        return self.quantizer.f_to_idx(features)

    def idx_to_img(self, indices):
        features = self.quantizer.idx_to_f(indices)
        features = self.post_quant_proj(features)
        img = self.decoder(features).clamp_(-1, 1)
        return img

    def img_to_reconstructed_img(self, image) -> torch.Tensor:
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_tokens, _, _, _ = self.quantizer(img_tokens)
        img_tokens = self.post_quant_proj(img_tokens)
        img_rec = self.decoder(img_tokens).clamp_(-1, 1)
        return img_rec


def get_meditok_args(img_size=256):
    return DotDict(dict(
        embed_dim=768,
        num_query=0,
        model='vitamin_large',
        img_size=img_size,
        drop_path=0,
        vocab_size=32768,
        vocab_width=64,
        vocab_norm=True,
        vq_beta=0.25,
        le=0.0,  # VQ entropy loss weight
        e_temp=0.01,
        num_codebooks=8,
        quant_proj='attn',
        grad_ckpt=True,
        device='cpu'
    ))


def build_meditok(ckpt_path, img_size=256):
    args = get_meditok_args(img_size)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model = MedITok(args)
    model.load_state_dict(state_dict)
    model = model.eval()
    del state_dict
    return model


if __name__ == '__main__':
    img_size = 256
    ckpt_path = '../weights/meditok/meditok_simple_v1.pth'
    net = build_meditok(ckpt_path, img_size=img_size)
    x = torch.randn((2, 3, img_size, img_size))
    with torch.no_grad():
        f = net.encode_image(x, verbose=True)

    print(x.shape, f.shape)
