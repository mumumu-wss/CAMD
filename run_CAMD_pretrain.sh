set -x

MASTER_ADDR="localhost"
NODE_RANK=0
NUM_NODES=1
NUM_GPUS=4

DATA_PATH="/mnt/cache/xlpr_sharedata/data_wss/VGGFACE2/data/vggface2_train"

PY_ARGS=${PY_ARGS:-""}

BASENAME=`basename ${0} .sh`
DIR=./output/pretrain/${BASENAME}_0.2pixel_test

mkdir -p ${DIR}

TOTAL_BATCH_SIZE=1024
BATCH_SIZE=$((TOTAL_BATCH_SIZE /  NUM_NODES / NUM_GPUS))

EPOCHS=400    # fast:50 total:400

BLR=3.125e-5    # blr:3.125e-5 lr:1.25e-4 lr=BLR*TOTAL_BATCH_SZIE/256

XI_1=1
XI_2=0.25
BETA_DIS=0.1

SELECTTOPK=2

DISLAY="2 5 8 11"   # "2 5 8 11"

MM=0.995        # mae: 0.996

MMS='cosine'    # mae:'const'

# crop range CROP_MIN-TO-CROP_MAX
CROP_MIN=0.08
CROP_MAX=0.60

CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /mnt/cache/wangsensen/wss/env/sim/bin/python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=11345 CAMD_pretrain.py \
    --model CAMD_vit_base_patch16 \
    --decoder_embed_dim 768 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs 5 \
    --crop_min ${CROP_MIN} \
    --crop_max ${CROP_MAX} \
    --with_blockwise_mask \
    --blockwise_num_masking_patches 118 \
    --blr ${BLR}  --weight_decay 0.05 \
    --mm ${MM} \
    --mmschedule ${MMS} \
    --clip_grad 1.0 \
    --loss_type 'CAMD' \
    --neg_weight 0.02 \
    --save_latest_freq 2 \
    --save_freq 50 \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --data_path ${DATA_PATH} \
    --select_top_k ${SELECTTOPK} \
    --xi_1 ${XI_1} \
    --xi_2 ${XI_2} \
    --beta_dis ${BETA_DIS} \
    --dis_lay ${DISLAY}\
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout.txt