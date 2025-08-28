#!/bin/bash
set -x

ROOT_DIR="/path/to/meditok"
WORKSPACE="${ROOT_DIR}/evaluation/generation"
cd ${WORKSPACE}/autoregressive/sample

SAMPLE_DIR="${ROOT_DIR}/outputs/generation/llamagen_meditok"
GPT_PATH="${ROOT_DIR}/weights/llamagen_meditok/meditok_c2i_gptb.pt"
VQ_PATH="${ROOT_DIR}/weights/meditok/meditok_simple_v1.pth"
TEST_DATA_PATH="${ROOT_DIR}/datasets/examples/c2i/medmnist-c2i-test.jsonl"

CUDA_VISIBLE_DEVICES="0" torchrun \
--nnodes=1 --nproc_per_node=1 --master_port=12345 \
sample_c2i.py \
--test_data_path ${TEST_DATA_PATH} \
--num_classes 6 \
--cls_token_num 1 \
--global_seed 0 \
--gpt_model GPT-B \
--sample_dir ${SAMPLE_DIR} \
--vq_ckpt ${VQ_PATH} \
--gpt_ckpt ${GPT_PATH} \
--temperature 1.0

