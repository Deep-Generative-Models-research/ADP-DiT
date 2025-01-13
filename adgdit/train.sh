#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29500  # 사용할 포트 지정
export PYTHONPATH=workspace/IndexKits:$PYTHONPATH

# Task and file settings
task_flag="dit_g_2"                                  # The task flag is used to identify folders.
index_file=dataset/AD3_meta/jsons/AD3_meta.json       # Index file for dataloader
results_dir=/workspace/mydata/adgdit/log_EXP_dit_g_2_AD3_meta              # Save root for results

# Training hyperparameters
batch_size=16                                        # Training batch size
image_size=256                                       # Training image resolution
grad_accu_steps=2                                    # Gradient accumulation
lr=0.0001                                            # Learning rate
ckpt_every=9999999                                   # Create a ckpt every a few steps.
ckpt_latest_every=9999999                            # Create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=100                               # Create a ckpt every a few epochs.
epochs=5000                                          # Total training epochs
warmup_num_steps=1932                                  # Warm-up steps
t_max=7728                                            # Steps per cosine cycle
eta_min=1e-5                                         # Minimum learning rate during decay
t_mult=2.0                                           # Multiplier for each cycle
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


    # --resume \
    # --resume-module-root ${resume_module_root} \
    # --resume-ema-root ${resume_ema_root} \


# OneCycle 스케쥴러
#    --lr_schedule OneCycle \                          # Specify the scheduler
#    --cycle-min-lr 0.00001 \                           # Minimum learning rate
#    --cycle-max-lr ${lr} \                             # Maximum learning rate
#    --cycle-first-step-size $((epochs / 2)) \          # Rising phase steps
#    --cycle-second-step-size $((epochs / 2)) \         # Falling phase steps

# Warmup 스케쥴러
    # --warmup-num-steps ${warmup_num_steps} \
    # --lr_schedule WarmupDecayLR \                    # Specify the scheduler
    # --total-num-steps ${total_steps} \                # Total number of steps for warmup and decay

# GPU 1장:
# 전체 update step = 337,500
# t_max=5700, warmup_steps=1425
# cycle_ratio = 5700 / 337,500 ≈ 0.0169
# warmup_ratio = 1425 / 337,500 ≈ 0.00422
# GPU 2장:
# 전체 update step = 84,000
# t_max_2 = 0.0169 × 84,000 ≈ 1420
# warmup_steps_2 = 0.00422 × 84,000 ≈ 350