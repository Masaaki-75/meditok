#!/bin/bash
set -x

ROOT_DIR="path/to/meditok"
WORKSPACE="${ROOT_DIR}/evaluation/generation"
cd ${WORKSPACE}

SAMPLE_DIR="${ROOT_DIR}/outputs/generation/inference/c2i"
GPT_PATH="${ROOT_DIR}/weights/generation/meditok_c2i_gptb.pt"
VQ_PATH="${ROOT_DIR}/weights/meditok/meditok.pth"


CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
--nnodes=1 --nproc_per_node=4 --master_port=12345 \
autoregressive/sample/sample_c2i.py \
--num-classes 6 \
--cls-token-num 1 \
--global-seed 0 \
--sample-dir ${SAMPLE_DIR} \
--vq-ckpt ${VQ_PATH} \
--gpt-ckpt ${GPT_PATH} \
--cfg-scale 1 \
#--vq-embed-path ${VQ_PATH}

