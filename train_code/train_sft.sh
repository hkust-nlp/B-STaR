export DATA_DIR=/path/to/your/data/directory
export MODEL_REPO= $DATA_DIR/checkpoints/Mistral-7B-v0.1

export OMP_NUM_THREADS=8


SFT_TRAIN_DATA=https://huggingface.co/datasets/AndrewZeng/math-trn-format/blob/main/math_format.json

# Please download this dataset to local folder
SFT_MODEL_SAVE_NAME=math_format_11k_mistral

torchrun --standalone --nproc_per_node=8 \
    train_sft.py \
    --do_train \
    --checkpoint_path $MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 1024 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 4 \
    --learning_rate 5e-6 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 3 \
    --dataset "$SFT_TRAIN_DATA" \
    --dataset_format "metamath" \
    --add_eos_to_marked_target \
    --save_strategy "steps" \
    --save_steps 25 \
    --optim_dtype bf16 \
    --save_total_limit 40 \
    --tensor_parallel_size 1 \
    --save_dir $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
    --resume_from_checkpoint