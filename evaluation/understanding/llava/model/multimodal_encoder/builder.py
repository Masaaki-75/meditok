import os
from .clip_encoder import CLIPVisionTower
from .meditok_encoder import MedITokVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    print(f'[multimodal_encoder/builder.py]------------> vision_tower_cfg: {vision_tower_cfg}')

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(f'[multimodal_encoder/builder.py]------------> vision_tower: {vision_tower}')
    
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    quantize = getattr(vision_tower_cfg, 'quantize', False)
    custom_encoder = getattr(vision_tower_cfg, 'custom_encoder', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if custom_encoder or quantize:
            print(f"[multimodal_encoder/builder.py]------------> Using MedITok!")
            return MedITokVisionTower(vision_tower, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')