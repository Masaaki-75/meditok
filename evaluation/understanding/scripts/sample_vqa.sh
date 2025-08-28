ROOT_DIR="/path/to/meditok"
WORKSPACE=${ROOT_DIR}/evaluation/understanding/llava
cd ${WORKSPACE}

IMAGE_FOLDER=${ROOT_DIR}/datasets/examples

python infer.py \
    --question_files ${ROOT_DIR}/datasets/examples/VQA/Slake/slake_test.jsonl \
    --image_folder ${IMAGE_FOLDER} \
    --model_path ${ROOT_DIR}/weights/llavamed_meditok \
    --answer_dir ${ROOT_DIR}/outputs/understanding/llavamed_meditok \
    --force_vision_tower_path ${ROOT_DIR}/weights/meditok/meditok_simple_v1.pth \
    --quantize --custom_encoder
