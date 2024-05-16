

python run_train.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 10 \
    --seed 42 \
    --model_name_or_path models/utc-base \
    --output_dir ./checkpoint/model_best \
    --dataset_path data/cail_mul_label_mul_classify \
    --max_seq_length 512  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model macro_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1 \
    --save_plm