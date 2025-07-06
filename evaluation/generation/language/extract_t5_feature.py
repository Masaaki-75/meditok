import os
import sys
sys.path.append('..')
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder


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


CAPTION_KEY = {
    'blip': 0,
    'llava': 1,
    'llava_first': 2,
}
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, jsonl_path, start=0, end=None, trunc_caption=False, rexgrad=True):
        img_path_list = []
        data = read_jsonl(jsonl_path)
        end = len(data) if end is None else end
        data = data[start:end]
        print(f"initializing from ({trunc_caption}, {rexgrad}): ", jsonl_path)
        for line_idx, line in enumerate(data):
            image_path = line['identifier']
            findings = line['findings']
            impression = line['impression']
            file_name = image_path.split('/')[-1]
            identifier = os.path.splitext(file_name)[0]

            #code_dir = image_path.split('/')[-1].split('.')[0]
            if len(impression) == 0 and len(findings) > 0:
                caption = findings
            else:
                if rexgrad:
                    caption = impression + ' ' + findings
                else:
                    caption = impression

            if trunc_caption:
                caption = caption.split('.')[0]
                if len(caption) <= 10:
                    caption = '.'.join(caption.split('.')[:2])
                if len(caption) <= 10:
                    caption = '.'.join(caption.split('.')[:3])

            caption = caption.strip()
            #img_path_list.append((caption, code_dir, line_idx))
            img_path_list.append((caption, identifier, line_idx))
        
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        caption, code_dir, code_name = self.img_path_list[index]
        return caption, code_dir, code_name


        
#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    #dist.init_process_group("nccl")
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.data_path, args.data_start, args.data_end, args.trunc_caption)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    for caption, code_dir, code_name in tqdm(loader):

        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu()
        save_path = os.path.join(args.save_dir, f'{code_dir[0]}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #save_path = os.path.join(args.save_dir, code_dir[0], '{}.pt')
        torch.save(x, save_path)
        #print(code_name.item())

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--data-end", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--caption-key", type=str, default='blip', choices=list(CAPTION_KEY.keys()))
    parser.add_argument("--trunc-caption", action='store_true', default=False)
    parser.add_argument("--t5-model-path", type=str, default=None)
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)

