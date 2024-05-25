#!/bin/sh
CURRENT_DIR=`pwd`

# accelerator="gpu"
# strategy="ddp"
# devices=1

train_data_path=data/cail_mul_label_mul_classify/test.txt
test_data_path=data/cail_mul_label_mul_classify/dev.txt

model_path=models/convert/utc_base
config_path=$model_path/config.json
vocab_path=$model_path

pretrained_model_path=$model_path/pytorch_model.bin
checkpoint_dir=models/checkpoint_dir
log_dir=logs

train_batch_size=32
test_batch_size=32
seq_length=512
learning_rate=1e-5

max_epochs=2

max_grad_norm=1
grad_accum_steps=1
eval_steps=10
logging_steps=10
save_checkpoint_steps=500

seed=42

lightning run model finetune.py \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --config_path $config_path \
    --vocab_path $vocab_path \
    --pretrained_model_path $pretrained_model_path \
    --checkpoint_dir $checkpoint_dir \
    --log_dir $log_dir \
    --train_batch_size $train_batch_size \
    --test_batch_size $test_batch_size \
    --seq_length $seq_length \
    --learning_rate $learning_rate \
    --max_epochs $max_epochs \
    --max_grad_norm $max_grad_norm \
    --grad_accum_steps $grad_accum_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --save_checkpoint_steps $save_checkpoint_steps \
    --seed $seed