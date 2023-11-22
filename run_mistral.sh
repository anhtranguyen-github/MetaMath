export MODEL_PATH='google/flan-t5-large'
export SAVE_PATH='./flan-t5-large-zalo-finetuned/'
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
export HF_TOKEN="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"
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train_math.py \
CUDA_VISIBLE_DEVICES=0 python3 train_zalo_mqa.py \
    --model_name_or_path $MODEL_PATH \
    --data_path ./data/train/processed_math_train.json \
    --data_length 1200 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 15 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "" \
    --save_strategy "steps" \
    --save_steps 100 \no
    --save_total_limit 0 \
    --learning_rate 2.5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --use_peft False
    #--optim paged_adamw_8bit \
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \

#python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
#python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
