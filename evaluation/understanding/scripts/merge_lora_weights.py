import sys
sys.path.append('..')
import argparse
from model.builder import load_pretrained_model
from mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        device_map='cpu',
        force_projector_path=args.projector,
        force_vision_tower_path=args.vision_tower,
        quantize=args.quantize,
        custom_encoder=args.custom_encoder
    )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path/to/your/trained/lora")
    parser.add_argument("--model_base", type=str, default="../../../weights/llava-med-v1.5-mistral-7b")
    parser.add_argument("--vision_tower", type=str, default=None)
    parser.add_argument("--projector", type=str, default=None)
    parser.add_argument("--quantize", action='store_true', default=False)
    parser.add_argument("--custom_encoder", action='store_true', default=False)

    args = parser.parse_args()
    model_path = args.model_path
    save_model_path = model_path.replace('lora', 'loramerged')
    args.save_model_path = save_model_path

    merge_lora(args)
