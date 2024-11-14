model='DiT-256/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed brdit/train_deepspeed.py ${params}  "$@"