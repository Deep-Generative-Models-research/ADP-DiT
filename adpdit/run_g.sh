model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "

# 추가: MASTER_PORT 환경 변수 사용
deepspeed --master_port=${MASTER_PORT} adpdit/train_deepspeed.py ${params} "$@"
