#!/bin/bash

# Define GPU and experiment-specific variables
run_experiment() {
    local gpu=$1
    local port=$2
    local task_flag=$3
    local index_file=$4
    local results_dir=$5
    local batch_size=$6  # Batch size is now passed as an argument

    echo "Starting experiment on GPU ${gpu} with MASTER_PORT=${port}, TASK_FLAG=${task_flag}, and BATCH_SIZE=${batch_size}"

    # Set environment variables for this experiment
    export CUDA_VISIBLE_DEVICES=${gpu}
    export MASTER_PORT=${port}

    # Training hyperparameters
    image_size=256
    grad_accu_steps=1
    warmup_num_steps=0
    lr=0.0001
    ckpt_every=9999999
    ckpt_latest_every=9999999
    ckpt_every_n_epoch=100
    epochs=1000

    # Run the experiment
    sh $(dirname "$0")/run_g.sh \
        --task-flag ${task_flag} \
        --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
        --predict-type v_prediction \
        --uncond-p 0 \
        --index-file ${index_file} \
        --random-flip \
        --lr ${lr} \
        --batch-size ${batch_size} \
        --image-size ${image_size} \
        --global-seed 999 \
        --grad-accu-steps ${grad_accu_steps} \
        --warmup-num-steps ${warmup_num_steps} \
        --use-flash-attn \
        --use-fp16 \
        --extra-fp16 \
        --results-dir ${results_dir} \
        --epochs ${epochs} \
        --ckpt-every ${ckpt_every} \
        --ckpt-latest-every ${ckpt_latest_every} \
        --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
        --log-every 50 \
        --deepspeed \
        --use-zero-stage 2 \
        --gradient-checkpointing \
        --cpu-offloading &
}

# Launch experiments with different batch sizes for each GPU
run_experiment 0 29500 "dit_g_2" "dataset/AD/jsons/AD.json" "./log_EXP_gpu0" 64
run_experiment 1 29501 "dit_g_2" "dataset/AD_meta/jsons/AD_meta.json" "./log_EXP_gpu1" 64
run_experiment 2 29502 "dit_XL_2" "dataset/AD_meta/jsons/AD_meta.json" "./log_EXP_gpu2" 64

# Wait for all experiments to finish
wait

echo "All experiments finished."
