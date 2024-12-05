export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=29503  # 사용할 포트 지정
export PYTHONPATH=/mnt/ssd/braintumor-DiT/IndexKits:$PYTHONPATH
# Task and file settings
task_flag="dit_g_2"                                  # the task flag is used to identify folders.
index_file=dataset/AD/jsons/AD.json                    # index file for dataloader
# resume_module_root=log_EXP/002-dit_XL_2/checkpoints/final.pt/mp_rank_00_model_states.pt # checkpoint root for model resume
# resume_ema_root=log_EXP/002-dit_XL_2/checkpoints/final.pt/zero_pp_rank_0_mp_rank_00_optim_states.pt     # checkpoint root for ema resume (필요한 경우 설정)
results_dir=./log_EXP                                  # save root for results

# Training hyperparameters
batch_size=64                                         # training batch size
image_size=256                                         # training image resolution
grad_accu_steps=1                                      # gradient accumulation
warmup_num_steps=0                                     # warm-up steps
lr=0.0001                                              # learning rate
ckpt_every=9999999                                     # create a ckpt every a few steps.
ckpt_latest_every=9999999                              # create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=100                                 # create a ckpt every a few epochs.
epochs=10000                                          # additional training epochs


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
    --t_max 1000 \
    --eta_min 1e-6 \
    --t_mult 2.0 \
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