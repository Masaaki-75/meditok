from .wds_image_dataset import get_wds_dataset
from .csv_image_dataset import get_csvimg_dataset


def load_data(args, epoch=0, iters=0, tokenizer=None):
    data = {}
    if args.train_data:
        if args.dataset_type == "csvimg":
            data["train"] = get_csvimg_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer)
        elif args.dataset_type == "wds":
            data["train"] = get_wds_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer)
        else:
            raise NotImplementedError(f"Unsupported training dataset type: {args.dataset_type}")
            
    if args.val_data:
        if args.dataset_type == "csvimg":
            data["val"] = get_csvimg_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer)
        elif args.dataset_type == "wds":
            data["val"] = get_wds_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer)
        else:
            raise NotImplementedError(f"Unsupported validation dataset type: {args.dataset_type}")

    return data