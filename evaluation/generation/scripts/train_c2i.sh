# !/bin/bash
set -x

ROOT_DIR="path/to/meditok"
WORKSPACE="${ROOT_DIR}/evaluation/generation"
cd ${WORKSPACE}

EPOCHS=300
LR=1e-4
BATCH_SIZE=128
VOCAB_SIZE=32768
CODE_DIR="${ROOT_DIR}/datasets/generation/medmnist-c2i-train.jsonl"
TOKENIZER_NAME="meditok"
RES_DIR="${ROOT_DIR}/outputs/generation"
EXP_NAME="meditok_c2i_gptb"
#VQ_PATH="${ROOT_DIR}/weights/meditok/meditok.pth"

#TORCH_LOGS="+dynamo"
#TORCHDYNAMO_VERBOSE=1
#CUDA_LAUNCH_BLOCKING=1 

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
--nnodes=1 --nproc_per_node=4 --master_port=12345 \
autoregressive/train/train_c2i.py \
--exp-name ${EXP_NAME} \
--code-dir ${CODE_DIR} \
--tokenizer-name ${TOKENIZER_NAME} \
--gpt-type "c2i" \
--dataset c2i_code \
--image-size 256 \
--epochs ${EPOCHS} \
--lr ${LR} \
--num-classes 6 \
--global-batch-size ${BATCH_SIZE} \
--vocab-size ${VOCAB_SIZE} \
--results-dir ${RES_DIR} \
#--vq-embed-path ${VQ_PATH}
#--no-compile
#--ema
