# Medical Image Synthesis with MedITok

This tutorial provides a simple guideline for building a medical image synthesis model with MedITok and GPT. For simplicity, we begin with the class-to-image generation.

## STEP 1: Data Preparation
Collect images with paired labels to construct the training and test sets, and construct the metadata file. 

We recommend `csv` or `jsonl` format for the metadata file. A training metadata file in JSONL format should be organized as follows: 
```javascript
[
    {"identifier": "/path/to/first/image", "code_identifier": "", "label": 0, "split": "train"}
    {"identifier": "/path/to/second/image", "code_identifier": "", "label": 1, "split": "train"}
    {"identifier": "/path/to/third/image", "code_identifier": "", "label": 0, "split": "train"}
    ...
]
```


## STEP 2: Image Tokenization using MedITok
We then perform offline image tokenization for the images collected to obtain the latent codes, using the following script. 
```sh
python batch_infer.py \
    --output_dir 'directory/to/save/the/latent/codes' \
    --pretrained_path 'path/to/the/pretrained/weights/of/visual/tokenizer' \
    --meta_path 'path/to/the/metadata/file' \
    --infer_type 'latent' \
    --image_size 256 \
    --batch_size 1 \
    --num_workers 8
```

Note that the input preprocessing in the `batch_infer.py` is for RGB images stored in `jpg`/`png` formats. If you are using other formats (like `nii` and `mhd`), please modify the preprocessing code accordingly in `datasets/simple_image_dataset.py`.

Running the command above will create image latent codes stored in `pt` files in the `output_dir`. The file name of each `pt` file follows the corresponding source image, except the postfix. After that, remember to update the `code_identifier` field in the metadata.

## STEP 3: Training and Inference
Once the latent codes are ready, modify the script `evaluation/generation/scripts/train_c2i.sh` for training a GPT model for class-to-image generation. Launch the training by
```sh
bash evaluation/generation/scripts/train_c2i.sh
```

When the training is done, modify the script `evaluation/generation/scripts/sample_c2i.sh` and start the inference by
```sh
bash evaluation/generation/scripts/sample_c2i.sh
```
