export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29500  # 사용할 포트 지정
export PYTHONPATH=/mnt/ssd/ADG-DiT/IndexKits:$PYTHONPATH

# Task and file settings
task_flag="dit_g_2"                                  # the task flag is used to identify folders.
index_file=dataset/AD2/jsons/AD2.json                    # index file for dataloader
results_dir=./log_EXP_dit_g_2_AD2                                  # save root for results

# Training hyperparameters
batch_size=64                                         # training batch size
image_size=256                                         # training image resolution
grad_accu_steps=2                                     # gradient accumulation
warmup_num_steps=2000                                     # warm-up steps
lr=0.0001                                              # learning rate
ckpt_every=9999999                                     # create a ckpt every a few steps.
ckpt_latest_every=9999999                              # create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=100                                 # create a ckpt every a few epochs.
epochs=2000                                          # additional training epochs
t_max=2000                                            # steps per cosine cycle
eta_min=1e-5                                          # minimum learning rate during decay
t_mult=2.0                                            # multiplier for each cycle

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
    --lr_schedule COSINE_ANNEALING_RESTARTS \
    --first-cycle-steps ${t_max} \
    --cycle-mult ${t_mult} \
    --gamma 0.9 \
    --warmup-steps ${warmup_num_steps} \
    --max-lr ${lr} \
    --min-lr ${eta_min} \
    --results-dir ${results_dir} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
    --log-every 50 \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    --cpu-offloading \
    "$@"




    # --resume \
    # --resume-module-root ${resume_module_root} \
    # --resume-ema-root ${resume_ema_root} \

# CosineAnnealingRestarts 스케쥴러
    # --lr_schedule COSINE_ANNEALING_RESTARTS \
    # --t_max 1000 \
    # --eta_min 1e-6 \
    # --t_mult 2.0 \


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