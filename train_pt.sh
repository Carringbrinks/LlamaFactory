FORCE_TORCHRUN=1 deepspeed --master_port=25678 --num_gpus=8 src/train.py \
    --model_name_or_path /data2/models/Qwen3-30B-A3B-Instruct-2507 \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset pretrain_data \
    --cutoff_len 38000 \
    --overwrite_cache \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 32 \
    --output_dir /data2/LLaMA-Factory/Train_Weights/v2_256k \
    --logging_steps 1 \
    --save_steps 0.4 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8  \
    --learning_rate 1.0e-4\
    --warmup_ratio 0.01 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --ddp_timeout 180000000 \
    --eval_steps 100 \
    --eval_strategy steps \
    --val_size 0.002 \
    --per_device_eval_batch_size 1 \
    --report_to swanlab \
    --run_name textileai_pretrain_38k \
    --disable_shuffling \
    # --max_samples 100000
    # --packing false
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    # --template qwen \
    # --save_only_model true \