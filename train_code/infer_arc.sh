RUN_NAME=arc_train_online_rft_mistral_temp0.4_7kdata
LOG_PATH=logs/${RUN_NAME}
QUERY_FILE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc_data/arc_test_data.jsonl
PROMPT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc_prompt"
mkdir -p $LOG_PATH

# for i in  {2500..4500..500}
# do
#     python vllm_infer_arc.py \
#         --input_file $QUERY_FILE \
#         --prompt_template_dir $PROMPT_DIR \
#         --shot_num 0 \
#         --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/checkpoints/$RUN_NAME/converted/ckpt$i \
#         --sample_num 1 \
#         --temperature 0 \
#         --top_k -1 \
#         --max_tokens 1024 \
#         --split test \
#         --save /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/gen.json \
#         --tensor-parallel-size 1 \
#         --cuda_ids 0 \
#         --cache_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/ARC/.cache \
#         2>&1 | tee -a $LOG_PATH/ckpt$i.log
    
#     python eval_arc_save_metrics.py \
#         --input_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/gen.json \
#         --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/metrics.json \
#         2>&1 | tee -a $LOG_PATH/ckpt$i.log
# done

RUN_NAME=Mistral-7B-Instruct-v0.1
for i in  {2500..2500..500}
do
    python vllm_infer_arc.py \
        --input_file $QUERY_FILE \
        --prompt_template_dir $PROMPT_DIR \
        --shot_num 1 \
        --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/modelzoo/Mistral-7B-Instruct-v0.1 \
        --sample_num 1 \
        --temperature 0 \
        --top_k -1 \
        --max_tokens 1024 \
        --split test \
        --save /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/gen.json \
        --tensor-parallel-size 1 \
        --cuda_ids 0 \
        --cache_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/ARC/.cache \
        2>&1 | tee -a $LOG_PATH/ckpt$i.log
    
    python eval_arc_save_metrics.py \
        --input_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/gen.json \
        --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/dynamic_ana/arc/results/${RUN_NAME}/ckpt$i/metrics.json \
        2>&1 | tee -a $LOG_PATH/ckpt$i.log
done