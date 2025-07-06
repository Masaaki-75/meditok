import os
import sys
import math
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import set_seed, logging, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from conversation import conv_templates, SeparatorStyle
from utils import disable_torch_init
from model import LlavaMistralForCausalLM
from mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

logging.set_verbosity_error()



def read_json(json_path, encoding='utf-8'):
    with open(json_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def write_json(data, json_path, write_mode='w', encoding='utf-8', ensure_ascii=False):
    with open(json_path, write_mode, encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)


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


def write_jsonl(dicts, jsonl_path, encoding='utf-8', write_mode='w', ensure_ascii=False):
    if not isinstance(dicts, (list, tuple)):
        dicts = [dicts,]
        
    with open(jsonl_path, write_mode, encoding=encoding) as f:
        for adict in dicts:
            f.write(json.dumps(adict, ensure_ascii=ensure_ascii) + '\n')
    

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    kwargs['torch_dtype'] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        **kwargs
    )
    
    image_processor = None

    if 'llava' in model_name.lower(): # or 'mistral' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        print(f"The vision tower is: \n{vision_tower}")
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        model.model.mm_projector.to(device=device, dtype=torch.float16)
        model.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def infer_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    question_files = args.question_files
    for question_file in question_files:
        print(f"Testing data from: {question_file}")
        base_name = os.path.basename(question_file)
        file_name, ext = os.path.splitext(base_name)
        assert ext == '.jsonl', f"Non-JSONL file is currently not supported."
        answer_file = os.path.join(args.answer_dir, file_name + f'_{model_name}{ext}')
        if os.path.exists(answer_file):
            print(f"Inference already done. Skipping: {answer_file}")
            continue

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        questions = read_jsonl(question_file)
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

        answers = []
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)
            ans = {"question_id": idx, "prompt": cur_prompt, "response": outputs, "model_id": model_name}
            answers.append(ans)

        print(f"Saving answers to: {answer_file}")
        write_jsonl(answers, answer_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../../../weights/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-files", type=str, nargs='+', default="../../../datasets/understanding/vqarad_test.jsonl")
    parser.add_argument("--answer-dir", type=str, nargs='+', default="../../../outputs/understanding/llavamed")
    parser.add_argument("--conv-mode", type=str, default="v1")  # "vicuna_v1"
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    question_files = args.question_files
    question_files = [question_files,] if isinstance(question_files, str) else question_files
    question_files = [
        '../../../datasets/understanding/vqarad_test.jsonl',
        '../../../datasets/understanding/slake_test.jsonl',
        '../../../datasets/understanding/slake_val.jsonl'
    ]
    args.question_files = question_files
    
    infer_model(args)
