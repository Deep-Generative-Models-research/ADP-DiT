#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29500
export PYTHONPATH=/data1/ADP-DiT/IndexKits:$PYTHONPATH

# 학습 관련 설정
task_flag="dit_g_2"
index_file="/data1/ADP-DiT/dataset/AD_meta/jsons/AD_meta.json"
results_dir="/data1/ADP-DiT/log_EXP_dit_g_2_AD_meta"

batch_size=8
image_size=256
grad_accu_steps=1
lr=0.0001
ckpt_every=9999999
ckpt_latest_every=9999999
ckpt_every_n_epoch=100
epochs=8888
warmup_num_steps=4000
t_max=26400
eta_min=1e-5
t_mult=2.0
gamma=0.9

# Run the training script
sh $(dirname "$0")/run_g.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear \
    --beta-start 0.00085 \
    --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --lr_schedule COSINE_ANNEALING_RESTARTS \
    --t_max ${t_max} \
    --eta_min ${eta_min} \
    --t_mult ${t_mult} \
    --warmup_steps ${warmup_num_steps} \
    --max_lr ${lr} \
    --min_lr ${eta_min} \
    --gamma ${gamma} \
    --results-dir ${results_dir} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
    --log-every 100 \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    --cpu-offloading \
    "$@"