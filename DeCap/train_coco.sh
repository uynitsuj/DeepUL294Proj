export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT=29501
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    train.py \
    --out_dir coco_model \
    --dataset coco \
    --epochs 10 \
    --local_rank 0

