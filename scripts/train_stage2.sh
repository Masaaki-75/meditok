#!/usr/bin
WORKSPACE="/path/to/your/meditok/repo"

cd ${WORKSPACE}

DATASET_TYPE="csvimg"
TRAIN_DATA="${WORKSPACE}/datasets/meta/meta2d_v2_train_example.csv"
TRAIN_ROOT="${WORKSPACE}/datasets/examples"
VAL_DATA="${WORKSPACE}/datasets/meta/meta2d_v2_test.csv"
VAL_ROOT="/path/to/data/root/"
CSV_IMG_KEY="identifier"
CSV_CAPTION_KEY="caption"

WORKERS=16
LOCAL_BS=64
IMG_SIZE=256
EPOCH=3
NUM_CODEBOOKS=8
VOCAB_SIZE=32768


EXP_NAME="meditok_s2_clipv01"
OUTPUT_DIR="${WORKSPACE}/outputs/ckpts/${EXP_NAME}"
RESUME_FROM="${WORKSPACE}/outputs/ckpts/meditok_s1_clipv01/ep2-iter65369.pth"
RECON_DIR="${WORKSPACE}/outputs/recon/s1"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

nnodes=1
nproc_per_node=4
master_port=20111

torchrun --nnodes=${nnodes} \
--nproc_per_node=${nproc_per_node} \
--master_port=${master_port} \
main.py \
--epoch $EPOCH --use_biomedclip True --vision_as_text False \
--lc 1.0 --lock_text True --ignore_text_params True --ct_bias 0 \
--local_bs $LOCAL_BS \
--vocab_size $VOCAB_SIZE \
--num_codebooks $NUM_CODEBOOKS \
--report_wandb False \
--model 'vitamin_large' \
--exp_name $EXP_NAME \
--img_size $IMG_SIZE \
--dataset_type $DATASET_TYPE \
--csv_img_key $CSV_IMG_KEY \
--csv_caption_key $CSV_CAPTION_KEY \
--train_root $TRAIN_ROOT \
--train_data $TRAIN_DATA \
--workers $WORKERS \
--vis_img_dir 'assets/vis_imgs/' \
--output_dir $OUTPUT_DIR \
--resume_from $RESUME_FROM \
--resume_net_only True

