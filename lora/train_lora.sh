export CUDA_VISIBLE_DEVICES=3
export MASTER_PORT=29503  # 사용할 포트 지정
export PYTHONPATH=/mnt/ssd/ADG-DiT/IndexKits:$PYTHONPATH

model='DiT-XL/2'                                                   # model type
task_flag="lora_adgdit_ema_rank64"                             # task flag
resume_module_root=/mnt/ssd/ADG-DiT/ADG-DiT_XL_2_ADoldversion/003-dit_XL_2/checkpoints/e4800.pt/zero_pp_rank_0_mp_rank_00_optim_states.pt   # resume checkpoint
index_file=/mnt/ssd/ADG-DiT/dataset/AD2_meta/jsons/AD2_meta.json                # the selected data indices
results_dir=./DiT-XL_2_AD2_meta_lora                                            # save root for results
batch_size=32                                                      # training batch size
image_size=256                                                   # training image resolution
grad_accu_steps=1                                                 # gradient accumulation steps
warmup_num_steps=0                                                # warm-up steps
lr=0.001                                                         # learning rate
ckpt_every=1000                                                    # create a ckpt every a few steps.
ckpt_latest_every=2000                                            # create a ckpt named `latest.pt` every a few steps.
rank=64                                                           # rank of lora
max_training_steps=14000                                          # Maximum training iteration steps

PYTHONPATH=./ deepspeed adgdit/train_deepspeed.py \
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



# python sample_t2i.py --infer-mode fa \
#     --prompt "Alzheimer Disease,Female, 84 years old, 24 months from first visit, Z-coordinate 150" \
#     --no-enhance --load-key ema \
#     --image-path "/workspace/dataset/AD_150/images/002_S_1280_2014-03-14_15_13_20.0_150.png" \
#     --lora-ckpt /workspace/log_EXP/003-lora_adgdit_ema_rank64/checkpoints/final.pt/adapter_model.safetensors




# python sample_t2i.py --infer-mode fa \
#     --prompt "Alzheimer, Female, Maternal Normal, Paternal Normal, Sibling Dementia Patient age: 71 years. Visited the clinic 4 months ago. Memory evaluations: MMD 100%, MML 100%, MMR 100%, MMO 100%, MMW 100%. MMSE score: 23.0 (mild impairment). CDR assessments: Memory 1.0, Orientation 1.0, Judgement 0.5, Communication 0.5, Home 0.5, Care 0.0" \
#     --no-enhance \
#     --image-path "/mnt/ssd/ADG-DiT/dataset/AD2/images/003_S_4644_2012-04-19_17_24_30.0_160.png" \
#     --dit-weight /mnt/ssd/ADG-DiT/ADG-DiT_XL_2_ADoldversion/003-dit_XL_2/checkpoints/e4800.pt/mp_rank_00_model_states.pt --load-key module \
#     --lora-ckpt /mnt/ssd/ADG-DiT/DiT-XL_2_AD2_meta_lora/001-lora_adgdit_ema_rank64/checkpoints/final.pt/adapter_model.safetensors 



# python sample_t2i.py --infer-mode fa \
#     --prompt "Alzheimer Disease, Female, 73 years old, 25 months from first visit" \
#     --no-enhance \
#     --image-path "/mnt/ssd/ADG-DiT/dataset/AD2/images/003_S_4644_2012-04-19_17_24_30.0_160.png" \
#     --dit-weight /mnt/ssd/ADG-DiT/DiT-XL_2_AD2_meta_lora/001-lora_adgdit_ema_rank64/checkpoints/latest.pt/adapter_model.safetensors  --load-key module 