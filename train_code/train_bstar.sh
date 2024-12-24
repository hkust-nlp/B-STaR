export MODEL_REPO= $DATA_DIR/checkpoints/Mistral-7B-v0.1
export OMP_NUM_THREADS=8

SFT_Midlle_Dir=/yourfolder/bstar_math
SFT_MODEL_SAVE_NAME=math_format_b_star_mistral
NUM_EPOCHS=9
NUM_ITERATIONS=9
NUM_GPUS=8

BASE_STEP=500
BASE_TARGET_SIZE=135000
GSM8K_TARGET_SIZE=81000

MATH_TARGET_SIZE=67500




# Check if the directory exists, if not, create it
if [ ! -d "$SFT_Midlle_Dir" ]; then
    mkdir -p "$SFT_Midlle_Dir"
    echo "Directory $SFT_Midlle_Dir created."
else
    echo "Directory $SFT_Midlle_Dir already exists."
fi

LOG_DIR=yourfolder/logs
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/train_bstar_log.txt


for ((iter=1; iter<=NUM_ITERATIONS; iter++))
do

    GEN_STEP=$(((iter-1) * BASE_STEP))

    sample_num=64

    # You should download following data

    input_data="https://huggingface.co/datasets/AndrewZeng/bstar-math-dev/blob/main/dynamic_ana_1k_withans_math.json"
    model_dir="$DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME/converted/ckpt$GEN_STEP"
    output_dir="$SFT_Midlle_Dir"
    tensor_parallel_size=1
    top_k=-1
    max_tokens=768
    wandb_project="vllm_gen"

    # select temperature
    start_temp=0.50
    end_temp=0.85
    temp_step=0.05

    gpu=0

    for temp in $(seq $start_temp $temp_step $end_temp); do
        output_file="$output_dir/dynamic_ana_1k_withans_iter${iter}_temp${temp}_sample${sample_num}_math.json"

        # For the first step, the model should be the first epoch checkpoint of SFT model

        # We will determine some configurations on small dev set
        CUDA_VISIBLE_DEVICES=$gpu python vllm_infer_auto.py \
            --input_data $input_data \
            --model_dir $model_dir \
            --sample_num $sample_num \
            --output_file $output_file \
            --tensor_parallel_size $tensor_parallel_size \
            --temperature $temp \
            --top_k $top_k \
            --max_tokens $max_tokens

        gpu=$(( (gpu + 1) % 9 ))
    done

    wait


    start_temp=0.9
    end_temp=1.2
    temp_step=0.05


    gpu=0


    for temp in $(seq $start_temp $temp_step $end_temp); do
        output_file="$output_dir/dynamic_ana_1k_withans_iter${iter}_temp${temp}_sample${sample_num}_math.json"

        CUDA_VISIBLE_DEVICES=$gpu python vllm_infer_auto.py \
            --input_data $input_data \
            --model_dir $model_dir \
            --sample_num $sample_num \
            --output_file $output_file \
            --tensor_parallel_size $tensor_parallel_size \
            --temperature $temp \
            --top_k $top_k \
            --max_tokens $max_tokens


        gpu=$(( (gpu + 1) % 9 ))
    done

    wait


    temps=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20)
    sample_nums=(64)

    # We will then use reward model to give reward



    for temp in "${temps[@]}"; do
        for sample_num in "${sample_nums[@]}"; do

            input_path="$output_dir/dynamic_ana_1k_withans_iter${iter}_temp${temp}_sample${sample_num}_math.json"
            output_path="$output_dir/dynamic_ana_1k_withans_infer4reward_iter${iter}_temp${temp}_sample${sample_num}_math.json"

            python convert4reward_auto_ground_sample.py \
                --input_path "$input_path" \
                --output_path "$output_path" \
                --sample_num $sample_num \
                --num_files -1   

            prompt_file="$output_path"
            reward_output_file="$output_dir/dynamic_ana_1k_withans_allreward_iter${iter}_temp${temp}_sample${sample_num}_math.json"

            torchrun --standalone --nproc_per_node=8 \
                inference_reward_llama3.py \
                --prompt_file "$prompt_file" \
                --output_file "$reward_output_file" \
                --batch_size 200 \
                --process_reward_with_answer \
                --tensor_parallel_size 1 \
                --checkpoint_path $DATA_DIR/checkpoints/Mistral-7B-v0.1/model.pth \
                --finetune_checkpoint_path $DATA_DIR/checkpoints/prm_model_mistral_sample_complete

        done
    done



    best_combination=$(python determine_hyper.py --temps 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 --sample_nums 64 \
    --input_path_template "$output_dir/dynamic_ana_1k_withans_allreward_iter${iter}_temp{temp}_sample{sample_num}_math.json" \
    --ref_path_template "$output_dir/dynamic_ana_1k_withans_infer4reward_iter${iter}_temp{temp}_sample{sample_num}_math.json" \
    --valid_sample_size 3550 \
    --iter $iter)


    temp=$(echo $best_combination | cut -d' ' -f1)
    sample_num=$(echo $best_combination | cut -d' ' -f2)


    export BEST_MATH_TEMP=$temp
    export BEST_MATH_SAMPLE_NUM=$sample_num


    echo "Best math temp: $BEST_MATH_TEMP" | tee -a $LOG_FILE
    echo "Best math sample_num: $BEST_MATH_SAMPLE_NUM" | tee -a $LOG_FILE


    for ((i=0; i<NUM_GPUS; i++))
    do
        CUDA_VISIBLE_DEVICES=$i python vllm_infer_auto.py \
            --input_data https://huggingface.co/datasets/AndrewZeng/math-bstar-sample/tree/main/output_ori/gsm8k_math_format_infer_part_$((i+1))_math.json \
            --model_dir /yourfolder/checkpoints/$SFT_MODEL_SAVE_NAME/converted/ckpt$GEN_STEP \
            --sample_num $BEST_MATH_SAMPLE_NUM \
            --output_file $SFT_Midlle_Dir/gsm8k_math_format_infer_part_$((i+1))_sample32_iter$((iter))_math.json \
            --tensor_parallel_size 1 \
            --temperature $BEST_MATH_TEMP \
            --top_k -1 \
            --max_tokens 768
    done
    wait


    python convert4reward_auto_ground_sample.py \
        --input_path $SFT_Midlle_Dir/gsm8k_math_format_infer_part_{}_sample32_iter$((iter))_math.json \
        --output_path $SFT_Midlle_Dir/gsm8k_math_format_infer_part_infer4reward_sample32_iter$((iter))_math.json \
        --num_files 8 \
        --sample_num $BEST_MATH_SAMPLE_NUM \
        2>&1 | tee -a $LOG_FILE


    torchrun --standalone --nproc_per_node=8 \
        inference_reward_llama3.py \
        --prompt_file $SFT_Midlle_Dir/gsm8k_math_format_infer_part_infer4reward_sample32_iter$((iter))_math.json \
        --output_file $SFT_Midlle_Dir/gsm8k_math_format_infer_part_allreward_sample32_iter$((iter))_math.json \
        --batch_size 200 \
        --process_reward_with_answer \
        --tensor_parallel_size 1 \
        --checkpoint_path $DATA_DIR/checkpoints/Mistral-7B-v0.1/model.pth \
        --finetune_checkpoint_path $DATA_DIR/checkpoints/prm_1to5_model_mistral_sample_complete \
        2>&1 | tee -a $LOG_FILE

    python metric_modiacc_auto.py \
        --input $SFT_Midlle_Dir/gsm8k_math_format_infer_part_allreward_sample32_iter$((iter))_math.json \
        --output $SFT_Midlle_Dir/gsm8k_math_format_infer_iter$((iter))_135k_math.json \
        --ref $SFT_Midlle_Dir/gsm8k_math_format_infer_part_infer4reward_sample32_iter$((iter))_math.json \
        --target_size $MATH_TARGET_SIZE \
        --correct_num 4 \
        --correct_ratio 0.9 \
        2>&1 | tee -a $LOG_FILE

    SFT_TRAIN_DATA=$SFT_Midlle_Dir/gsm8k_math_format_infer_iter$((iter))_135k_math.json

    END_STEP=$((iter * BASE_STEP))

    if [ $iter -eq 1 ]; then
        torchrun --standalone --nproc_per_node=$NUM_GPUS \
            train_sft_step.py \
            --do_train \
            --checkpoint_path $MODEL_REPO/model.pth \
            --sft_checkpoint_path $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
            --source_max_len 768 \
            --target_max_len 768 \
            --total_max_len 1024 \
            --per_device_train_batch_size 16 \
            --micro_train_batch_size 8 \
            --learning_rate 4.12e-6 \
            --lr_eta_min 2e-7 \
            --num_train_epochs $NUM_EPOCHS \
            --dataset "$SFT_TRAIN_DATA" \
            --dataset_format "metamath" \
            --add_eos_to_marked_target \
            --save_strategy "steps" \
            --save_steps 500 \
            --optim_dtype bf16 \
            --save_total_limit 40 \
            --tensor_parallel_size 1 \
            --end_step $END_STEP \
            --save_dir $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
            2>&1 | tee -a $LOG_FILE
    else
        torchrun --standalone --nproc_per_node=$NUM_GPUS \
            train_sft_step.py \
            --do_train \
            --checkpoint_path $MODEL_REPO/model.pth \
            --sft_checkpoint_path $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
            --source_max_len 768 \
            --target_max_len 768 \
            --total_max_len 1024 \
            --per_device_train_batch_size 16 \
            --micro_train_batch_size 8 \
            --learning_rate 4.12e-6 \
            --lr_eta_min 2e-7 \
            --num_train_epochs $NUM_EPOCHS \
            --dataset "$SFT_TRAIN_DATA" \
            --dataset_format "metamath" \
            --add_eos_to_marked_target \
            --save_strategy "steps" \
            --save_steps 500 \
            --optim_dtype bf16 \
            --save_total_limit 40 \
            --tensor_parallel_size 1 \
            --end_step $END_STEP \
            --save_dir $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
            --resume_from_checkpoint \
            2>&1 | tee -a $LOG_FILE
    fi

    python convert_auto.py \
        --checkpoint_dir $DATA_DIR/checkpoints/$SFT_MODEL_SAVE_NAME \
        --pretrain_name $MODEL_REPO \
        --tokenizer_name $MODEL_REPO \
        2>&1 | tee -a $LOG_FILE
done

