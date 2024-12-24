export DATA_DIR=/path/to/your/data/directory

export MODEL_REPO= $DATA_DIR/checkpoints/Mistral-7B-v0.1
export OMP_NUM_THREADS=4


RM_DATA=train_prm_math_shepherd_mistral.json
RM_MODEL_SAVE_NAME=prm_model_mistral_sample_complete

torchrun --standalone --nproc_per_node=8 \
    train_rm_pointwise.py \
    --do_train \
    --checkpoint_path $MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 1024 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 32 \
    --learning_rate 2e-6 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 2 \
    --dataset "$RM_DATA" \
    --dataset_format "prm-v4" \
    --save_strategy epoch \
    --save_total_limit 5 \
    --train_on_every_token \
    --tensor_parallel_size 1 \
    --save_only_model True \
    --optim_dtype bf16 \
    --save_dir $DATA_DIR/checkpoints/$RM_MODEL_SAVE_NAME \
    --resume_from_checkpoint