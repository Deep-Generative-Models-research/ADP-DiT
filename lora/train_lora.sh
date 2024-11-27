model='DiT-256/2'                                                   # model type
task_flag="lora_brdit_ema_rank64"                             # task flag
resume_module_root=/workspace/log_EXP/001-dit_256_2/checkpoints/e0500.pt/zero_pp_rank_0_mp_rank_00_optim_states.pt    # resume checkpoint
index_file=/workspace/dataset/AD_150/jsons/AD_150.json                # the selected data indices
results_dir=./log_EXP                                             # save root for results
batch_size=32                                                      # training batch size
image_size=256                                                   # training image resolution
grad_accu_steps=1                                                 # gradient accumulation steps
warmup_num_steps=0                                                # warm-up steps
lr=0.0001                                                         # learning rate
ckpt_every=100                                                    # create a ckpt every a few steps.
ckpt_latest_every=2000                                            # create a ckpt named `latest.pt` every a few steps.
rank=64                                                           # rank of lora
max_training_steps=2000                                          # Maximum training iteration steps

PYTHONPATH=./ deepspeed brdit/train_deepspeed.py \
    --task-flag ${task_flag} \
    --model ${model} \
    --training-parts lora \
    --rank ${rank} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --lr ${lr} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --uncond-p-t5 0 \
    --index-file ${index_file} \
    --random-flip \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --max-training-steps ${max_training_steps}\
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 50 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --qk-norm \
    --rope-img base512 \
    --rope-real \
    "$@"


'''
python sample_t2i.py --infer-mode fa \
    --prompt "Alzheimer Disease,Female, 84 years old, 24 months from first visit, Z-coordinate 150" \
    --no-enhance --load-key ema \
    --image-path "/workspace/dataset/AD_150/images/002_S_1280_2014-03-14_15_13_20.0_150.png" \
    --lora-ckpt /workspace/log_EXP/003-lora_brdit_ema_rank64/checkpoints/final.pt/adapter_model.safetensors

'''

'''
python sample_t2i.py --infer-mode fa \
    --prompt "Alzheimer Disease, Female, 84 years old, 24 months from first visit, Z-coordinate 150" \
    --no-enhance \
    --image-path "/workspace/dataset/AD_150/images/002_S_1280_2014-03-14_15_13_20.0_150.png" \
    --dit-weight log_EXP/001-dit_256_2/checkpoints/e0500.pt \
    --lora-ckpt log_EXP/002-lora_brdit_ema_rank64/checkpoints/final.pt/adapter_model.safetensors \
    --load-key module
'''
