import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.meditok import build_meditok
from datasets.simple_image_dataset import SimpleImageDataset


def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


def check_file_ok(path: str):
    try:
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            _ = Image.open(path)
        else:
            pass
            #_ = torch.load(path, map_location='cpu')
        return True
    except Exception:
        return False


def save_image(
    img: torch.Tensor, 
    save_path: str, 
    mode='RGB',
    input_min=-1,
    input_max=-1
):
    assert img.shape[0] == 1, f"Batch size must be 1, got shape {img.shape}."
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = stable_linear_transform(
        img, 
        input_min=input_min, 
        input_max=input_max, 
        output_min=0,
        output_max=255,
        do_clip=True
    )
    
    # [B, C, H, W] -> [B, H, W, C]
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0]).convert(mode)
    img.save(save_path)


@torch.no_grad()
def infer_model(
    model, 
    meta_path, 
    output_dir, 
    dataset_dir=None, 
    infer_type='image', 
    image_size=256, 
    model_input_channel=3,
    model_input_min=-1,
    model_input_max=1,
    num_workers=4, 
    batch_size=1,
    device='cuda'
):
    os.makedirs(output_dir, exist_ok=True)
    ext = '.png' if infer_type == 'image' else '.pt'
    
    dataset = SimpleImageDataset(
        meta_path, 
        img_key='identifier', 
        root_dir=dataset_dir,
        image_size=image_size,
        output_channel=model_input_channel,
        output_min=model_input_min,
        output_max=model_input_max,
    )
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    
    for bidx, (imgs, img_paths) in tqdm(enumerate(loader), total=len(loader), desc=f"Inferring {infer_type}"):
        
        save_paths = []
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            filename = os.path.splitext(basename)[0]
            identifier = filename + ext
            save_path = os.path.join(output_dir, identifier)
            save_paths.append(save_path)
        
        do_skip = all([os.path.exists(p) and check_file_ok(p) for p in save_paths])
        
        if not do_skip:
            imgs = imgs.to(device)
            if infer_type == 'image':
                code_idx = model.img_to_idx(imgs)
                results = model.idx_to_img(code_idx)
                for i in range(results.shape[0]):
                    save_image(results[i], save_paths[i])
                    
            elif infer_type == 'vector':
                results = model.encode_image(imgs).detach().squeeze().cpu()
                for i in range(results.shape[0]):
                    torch.save(results[i], save_paths[i])
                    
            elif infer_type == 'latent':
                results = model.encode_image_vq(imgs).detach()
                results = results.permute(0, 2, 1).reshape(results.shape[0], 64, 16, 16).squeeze().cpu()
                for i in range(results.shape[0]):
                    torch.save(results[i], save_paths[i])
                
            else:
                raise NotImplementedError(f"MedITok only supports infer_type in ['image', 'vector', 'latent'], but got {infer_type}.")


def parse_args():
    parser = argparse.ArgumentParser(description='make my life easier') 
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for inference results.')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the model weights')
    parser.add_argument('--meta_path', type=str, default=None, help='Path to metadata (in a csv/jsonl file)')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Directory for the ')
    parser.add_argument('--infer_type', type=str, default='image', help='output type for inference.')
    parser.add_argument('--model_input_min', type=float, default=-1)
    parser.add_argument('--model_input_min', type=float, default=-1)
    parser.add_argument('--model_input_channel', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # EXAMPLE USAGE: python batch_infer.py --output_dir 'xxx' --pretrained_path 'xxx' --meta_path 'xxx' --infer_type 'latent'

    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_meditok(args.pretrained_path, device=device).eval()
    infer_model(
        model,
        args.meta_path,
        args.output_dir, 
        dataset_dir=args.dataset_dir, 
        infer_type=args.infer_type, 
        image_size=args.image_size, 
        model_input_channel=args.model_input_channel,
        model_input_min=args.model_input_min,
        model_input_max=args.model_input_max,
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        device=device
    )
