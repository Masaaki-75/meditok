from dataset.c2i import build_c2i_code


def build_dataset(args, **kwargs):
    if args.dataset == 'c2i_code':
        return build_c2i_code(args, **kwargs)
    
    raise NotImplementedError(f'Dataset {args.dataset} is not supported')