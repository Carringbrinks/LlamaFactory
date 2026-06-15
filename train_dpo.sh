# export NCCL_P2P_LEVEL=NVL

FORCE_TORCHRUN=1 deepspeed --master_port=25678 --num_gpus=8 src/train.py \
    --model_name_or_path /data2/LLaMA-Factory/Train_Weights/v2_256k/checkpoint-1892 \
    --trust_remote_code true \
    --stage dpo \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset rlhf_textileai,rlhf_textileai_general,rlhf_identy,rlhf_general \
    --template qwen3_nothink \
    --cutoff_len 8192 \
    --overwrite_cache \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 32 \
    --output_dir /data2/LLaMA-Factory/Train_Weights/dpo_weights \
    --logging_steps 1 \
    --save_steps 0.2 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8  \
    --learning_rate 1e-5\
    --warmup_ratio 0.01 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --ddp_timeout 180000000 \
    --eval_steps 100 \
    --eval_strategy steps \
    --val_size 0.005 \
    --per_device_eval_batch_size 4 \
    --report_to swanlab \
    --run_name textileai_dpo_test \
    # --max_samples 1000
    # --run_name qwen2.5_1.5b_sft \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    
  
    
