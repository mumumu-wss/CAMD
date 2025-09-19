set -x

MASTER_ADDR="localhost"
NODE_RANK=0
NUM_NODES=1
NUM_GPUS=2
MASTER_PORT=12456

# check-point
CKPT_PATH=Data/pretrain_checkpoint/pretrain_checkpoint/CAMD_99ep.pth

# fine-tune_data_path
DATA_PATH="/Data/down_task/fer/ferplus/data"   # RAF-DB:/Data/down_task/fer/RAF-DB/basic/

# data_name
#DATA_NAME="rafdb"    # ferplus

#NUM_CLASSES=8   # 'rafdb': 7, "fer2013": 7

#EPOCH=85 # rafdb:85 ferplus:80

PY_ARGS=${PY_ARGS:-""}

TOTAL_BATCH_SIZE=256
BATCH_SIZE=$((TOTAL_BATCH_SIZE /  NUM_NODES / NUM_GPUS))

BLR=2.5e-4

FEWSHOT_RATE=0.01

DIR=./exp/CAMD_${FEWSHOT_RATE}FEWSHOT_${DATA_NAME}_99ep_ep${EPOCH}_bs${BATCH_SIZE}_blr${BLR}

mkdir -p ${DIR}

CUDA_VISIBLE_DEVICES=0,1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    CAMD_finetune.py \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --batch_size ${BATCH_SIZE} \
    --model vit_base_patch16 \
    --few_shot_ratio ${FEWSHOT_RATE} \
    --finetune ${CKPT_PATH} \
    --epochs ${EPOCH} \
    --task fer \
    --blr ${BLR} --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25\
    --dist_eval --data_path ${DATA_PATH} \
    --data_name ${DATA_NAME} \
    --nb_classes ${NUM_CLASSES} \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout.txt