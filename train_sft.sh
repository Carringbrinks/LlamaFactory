export HF_DATASETS_CACHE="/data2/LLaMA-Factory/.cache"

FORCE_TORCHRUN=1 deepspeed --master_port=25678 --num_gpus=8 src/train.py \
    --model_name_or_path /data2/LLaMA-Factory/Train_Weights/v2_38k/checkpoint-1892 \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset sft_textileai,sft_identy,sft_general,sft_textileai_general \
    --template qwen3_nothink \
    --cutoff_len 8192 \
    --overwrite_cache \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 32 \
    --output_dir /data2/LLaMA-Factory/Train_Weights/sft_weights \
    --logging_steps 1 \
    --save_steps 0.6 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8  \
    --learning_rate 1e-5\
    --warmup_ratio 0.01 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 100 \
    --eval_strategy steps \
    --val_size 0.005 \
    --per_device_eval_batch_size 4 \
    --report_to swanlab \
    --run_name textileai_sft_8k \
    --save_only_model true
    # --run_name qwen2.5_1.5b_sft \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    
  
    
