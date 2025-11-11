import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from models import init_weights
from models.meditok import MedITok, build_meditok
from local_openclip.constants import BIOMEDCLIP_CKPT
from local_openclip.factory import create_model_from_pretrained


class TokenizerWrapper(nn.Module):
    def __init__(self, args, core_class=MedITok):
        super().__init__()
        self.core: MedITok = core_class(args)
        self.drop_alignment =self.core.drop_alignment
        text_cfg = {
            "width": args.text_width,
            "heads": args.text_heads,
            "layers": args.text_layers,
            "vocab_size": args.text_vocab_size,
            "context_length": args.text_context_length,
        }

        self.text_no_grad = False

        if self.drop_alignment:
            self.text_encoder = self.text_fc_norm = self.text_projection = None
        else:
            biomedclip, preprocess = create_model_from_pretrained('BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', BIOMEDCLIP_CKPT)
            self.context_length = biomedclip.text.context_length  # 512
            self.vocab_size = biomedclip.text.vocab_size          # 30522
            self.maybe_record_function = nullcontext

            self.vision_as_text = args.vision_as_text
            if args.vision_as_text:  # BiomedCLIP-vision as text encoder!
                self.text_encoder = biomedclip.visual.trunk  # [batch_size, 768]
                self.text_fc_norm = nn.LayerNorm(self.text_encoder.embed_dim, eps=1e-6)
                self.text_projection = nn.Linear(self.text_encoder.embed_dim, args.embed_dim)
            else:
                self.text_encoder = biomedclip.text #BiomedCLIP(args.embed_dim, text_cfg)
                # BiomedCLIP final outputs 512 dim (intermediate outputs 768, but i don't know how to pool)
                self.text_fc_norm = nn.LayerNorm(self.text_encoder.output_dim, eps=1e-6)
                self.text_projection = nn.Linear(self.text_encoder.output_dim, args.embed_dim)
            
            self.text_encoder.set_grad_checkpointing(args.grad_ckpt)


    def forward(self, img, text=None, ret_usages=False):
        img_tokens = self.core.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            img_tokens = self.core.quant_proj(img_tokens)
            img_tokens, vq_loss, entropy_loss, usages = self.core.quantizer(img_tokens)
            img_tokens = self.core.post_quant_proj(img_tokens)
        img_rec = self.core.decoder(img_tokens).float()

        if self.drop_alignment:
            clip_text = clip_visual = exp_logit = None
        else:
            clip_visual = img_tokens.mean(dim=1)
            clip_visual = self.core.projection(self.core.fc_norm(clip_visual))
            clip_visual = F.normalize(clip_visual, dim=-1)

            if self.vision_as_text:
                clip_text = self.aux_encode_image(img, normalize=True)
            else:
                clip_text = self.encode_text(text, normalize=True) if text is not None else None

            exp_logit = self.core.logit_scale.exp()

        output_dict = {
            "img_rec": img_rec,
            "vq_loss": vq_loss,
            "entropy_loss": entropy_loss,
            "codebook_usages": usages,
            "clip_image_features": clip_visual,
            "clip_text_features": clip_text,
            "logit_scale": exp_logit,
        }
        return output_dict
    
    def encode_image_vq(self, image):
        return self.core.encode_image_vq(image)

    def encode_image(self, image, normalize=False):
        return self.core.encode_image(image, normalize=normalize)
    
    def img_to_idx(self, img):
        return self.core.img_to_idx(img)
    
    def idx_to_img(self, indices):
        return self.core.idx_to_img(indices)
    
    def img_to_reconstructed_img(self, image):
        return self.core.img_to_reconstructed_img(image)

    def aux_encode_image(self, image, normalize=False):
        assert self.vision_as_text and (not self.drop_alignment)
        if image.shape[-1] != 224:  # BiomedCLIP requires 224x224 input images
            image = F.interpolate(image, size=(224, 224))
        
        if self.text_no_grad:
            with torch.no_grad():
                features = self.text_encoder(image).detach()
        else:
            features = self.text_encoder(image)
        features = self.text_projection(self.text_fc_norm(features))
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        assert not self.drop_alignment
        features = self.text_encoder(text)
        features = self.text_projection(self.text_fc_norm(features))
        return F.normalize(features, dim=-1) if normalize else features

    def lock_text_tower(
        self, 
        freeze_logit_scale=False,
        **kawrgs
    ):
        if hasattr(self.text_encoder, 'parameters'):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        if hasattr(self.core, 'logit_scale'):
            if freeze_logit_scale:
                self.core.logit_scale.requires_grad = False
            
        self.text_no_grad = True
    
    def lock_visual_projector(self):
        if hasattr(self.core, 'projection'):
            for p in self.core.projection.parameters():
                p.requires_grad = False
        
        if hasattr(self.core, 'fc_norm'):
            for p in self.core.fc_norm.parameters():
                p.requires_grad = False


def build_meditok_wrapper(args):
    device = getattr(args, 'device', 'cpu')
    model = TokenizerWrapper(args, core_class=MedITok).to(device)

    # init_weights(model.core.encoder, args.vae_init)
    init_weights(model.core.decoder, args.vae_init)
    init_weights(model.core.quant_proj, args.vae_init)
    init_weights(model.core.post_quant_proj, args.vae_init)
    model.core.quantizer.init_vocab(args.vocab_init)
    return model