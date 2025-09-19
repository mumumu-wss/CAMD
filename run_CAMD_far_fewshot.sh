set -x

MASTER_ADDR="localhost"
NODE_RANK=0
NUM_NODES=1
NUM_GPUS=2
MASTER_PORT=11345

# check-point
CKPT_PATH=Data/pretrain_checkpoint/pretrain_checkpoint/CAMD_99ep.pth

NUM_CLASSES=40

# fine-tune_data_path
#DATA_PATH="/Data/down_task/far/LFWA/"   # Celeba:/Data/down_task/far/CelebA/

# fine-tune_task_name
#DATA_NAME="lfwa"   # celeba

EPOCH=500

PY_ARGS=${PY_ARGS:-""}

TOTAL_BATCH_SIZE=256
BATCH_SIZE=$((TOTAL_BATCH_SIZE /  NUM_NODES / NUM_GPUS))

BLR=2.5e-4    # lr=BLR*TOTAL_BATCH_SZIE/256

FEWSHOT_RATE=0.01

DIR=./exp/CAMD_${FEWSHOT_RATE}FEWSHOT_${DATA_NAME}_99ep_ep${EPOCH}_bs${BATCH_SIZE}_blr${BLR}

CUDA_VISIBLE_DEVICES=0,1

mkdir -p ${DIR}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    CAMD_finetune.py \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --batch_size ${BATCH_SIZE} \
    --model vit_base_patch16 \
    --few_shot_ratio ${FEWSHOT_RATE} \
    --finetune ${CKPT_PATH} \
    --epochs ${EPOCH} \
    --task far \
    --blr ${BLR} --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25\
    --dist_eval --data_path ${DATA_PATH} \
    --data_name ${DATA_NAME} \
    --nb_classes ${NUM_CLASSES} \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout.txt