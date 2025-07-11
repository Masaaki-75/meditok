#!/bin/bash
set -x

ROOT_DIR="path/to/meditok/repo"
WORKSPACE="${ROOT_DIR}/evaluation/generation"
cd ${WORKSPACE}

SAMPLE_DIR="${ROOT_DIR}/outputs/generation/inference/c2i"
GPT_PATH="${ROOT_DIR}/weights/generation/meditok_c2i_gptb.pt"
VQ_PATH="${ROOT_DIR}/weights/meditok/meditok.pth"
TEST_DATA_PATH="${ROOT_DIR}/datasets/generation/medmnist-c2i-test.jsonl"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
--nnodes=1 --nproc_per_node=4 --master_port=12345 \
autoregressive/sample/sample_c2i.py \
--test-data-path ${TEST_DATA_PATH} \
--num-classes 6 \
--cls-token-num 1 \
--global-seed 0 \
--gpt-model GPT-B \
--sample-dir ${SAMPLE_DIR} \
--vq-ckpt ${VQ_PATH} \
--gpt-ckpt ${GPT_PATH} \
--cfg-scale 1 \
--temperature 1.0 \
#--vq-embed-path ${VQ_PATH}

