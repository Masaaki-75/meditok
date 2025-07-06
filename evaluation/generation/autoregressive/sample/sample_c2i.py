import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import torch.nn.functional as F
import torch.distributed as dist

import os
import sys
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

sys.path.append('../../../..')
from models.meditok import build_meditok
from evaluation.generation.autoregressive.models.gpt_c2i import GPT_models
from evaluation.generation.autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from evaluation.generation.utils.unitok_config import Args


def read_jsonl(file_path, encoding='utf-8', skip_error=False):
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))  # Convert each JSONL line to a dictionary
            except Exception as err:
                print(f"Error when loading Line {idx} in {file_path}: {err}")
                if skip_error:
                    continue
                else:
                    raise err
    return data


def divide_list_with_overlap(a, group_len, overlap=0):
    if group_len <= 0 or overlap < 0 or group_len <= overlap:
        raise ValueError("group_len must be positive and greater than overlap.")

    groups = []
    start = 0

    while start < len(a):
        end = start + group_len
        groups.append(a[start:end])
        start += group_len - overlap
    return groups



MODALITY_MAPPING = {
    0: 'microscopy',
    1: 'ultrasound',
    2: 'x-ray',
    3: 'dermoscopy',
    4: 'histopathology',
    5: 'fundus-photography'
}

def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = build_meditok(args.vq_ckpt)

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = 16
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        num_codebooks=args.num_codebooks,
        n_output_layer=args.num_output_layer,
        vq_embed_path=args.vq_embed_path,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:  # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile")

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    vq_name = args.vq_ckpt.split('/')[-2]
    if args.vq_embed_path is not None:
        vq_name = vq_name + '-embed'
    
    ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{vq_name}-{model_string_name}-{ckpt_string_name}-" \
                  f"temp-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    test_data_path = args.test_data_path
    test_data = read_jsonl(test_data_path)
    
    pbar = divide_list_with_overlap(test_data, group_len=args.per_proc_batch_size)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for batch in pbar:
        #c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        token_id_paths = [b['identifier'] for b in batch]
        identifiers = [p.split('/')[-1].replace('.pt', '') for p in token_id_paths]
        labels = [int(b['label']) for b in batch]
        c_indices = torch.tensor(labels, device=device)
        modalities = [MODALITY_MAPPING[c] for c in labels]
        save_paths = [f"{sample_folder_dir}/{modalities[i]}/{identifiers[i]}.png" for i in range(len(batch))]
        if all([os.path.exists(p) for p in save_paths]):
            print(f"Already generated. Skipping: {save_paths}")
            continue

        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, args.num_codebooks,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
        )

        # samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        samples = vq_model.idx_to_img(index_sample)
        if samples.shape[-1] != args.image_size or samples.shape[-2] != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size, args.image_size), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            identifier = identifiers[i]
            save_path = f"{sample_folder_dir}/{modalities[i]}/{identifier}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(sample).save(save_path)
        
        if args.copy_input:
            for i in range(len(token_id_paths)):
                token_id_path = token_id_paths[i]
                identifier = token_id_path.split('/')[-1].replace('.pt', '')
                save_path = f"{args.sample_dir}/input/{modalities[i]}/{identifier}.png"
                if not os.path.exists(save_path):
                    index_sample = torch.load(token_id_path, map_location='cuda')
                    samples = vq_model.idx_to_img(index_sample.unsqueeze(0))
                    if samples.shape[-1] != args.image_size or samples.shape[-2] != args.image_size:
                        samples = F.interpolate(samples, size=(args.image_size, args.image_size), mode='bicubic')
                    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    Image.fromarray(samples[0]).save(save_path)

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i",
                        help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=32768, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--num-output-layer", type=int, default=1)
    parser.add_argument("--vq-embed-path", type=str, default=None)
    parser.add_argument("--copy-input", default=False, action='store_true')
    parser.add_argument("--test-data-path", type=str, default='../../../../datasets/generation/medmnist-c2i-test.jsonl')
    args = parser.parse_args()
    main(args)