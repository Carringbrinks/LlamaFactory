
deepspeed --master_port=25678 --num_gpus=8 src/train.py \
    --model_name_or_path /home/cbsu/huggingface/Qwen2.5-14B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset Steel_multiple_choice_questions,Steel_Quiz \
    --template qwen \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 24 \
    --output_dir /home/cbsu/Pyproject/LLaMA-Factory/Train_Weights/qwen2.5-14b-sft \
    --logging_steps 1 \
    --save_steps 0.5 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8  \
    --learning_rate 1e-5\
    --warmup_steps 20 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 50 \
    --eval_strategy steps \
    --val_size 0.01 \
    --per_device_eval_batch_size 2 \
    --save_only_model true \
    --report_to swanlab \
    --run_name qwen2.5_14b \
    # --run_name qwen2.5_1.5b_sft \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    
  
    
