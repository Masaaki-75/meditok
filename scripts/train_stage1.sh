#!/usr/bin
WORKSPACE="/path/to/your/meditok/repo"

cd ${WORKSPACE}

DATASET_TYPE="csvimg"
TRAIN_DATA="${WORKSPACE}/datasets/vae/meta2d_train.csv"
TRAIN_ROOT="/path/to/data/root/"
VAL_DATA="${WORKSPACE}/datasets/vae/meta2d_test.csv"
VAL_ROOT="/path/to/data/root/"
CSV_IMG_KEY="identifier"
CSV_CAPTION_KEY="caption"

WORKERS=8
LOCAL_BS=64
IMG_SIZE=256
EPOCH=3
NUM_CODEBOOKS=8
VOCAB_SIZE=32768

EXP_NAME="meditok_s1_clipv01"
OUTPUT_DIR="${WORKSPACE}/outputs/unitok/${EXP_NAME}"
#RESUME_FROM="${WORKSPACE}/weights/unitok/unitok_tokenizer.pth"
RECON_DIR="${WORKSPACE}/outputs/recon/s1"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

nnodes=1
nproc_per_node=4
master_port=20165

torchrun --nnodes=${nnodes} \
--nproc_per_node=${nproc_per_node} \
--master_port=${master_port} \
main.py \
--epoch $EPOCH --use_biomedclip True --vision_as_text True \
--lc 0.1 --lock_text True --ct_bias 1024 \
--local_bs $LOCAL_BS \
--vocab_size $VOCAB_SIZE \
--num_codebooks $NUM_CODEBOOKS \
--report_wandb False \
--eval_per_epoch 10 \
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

