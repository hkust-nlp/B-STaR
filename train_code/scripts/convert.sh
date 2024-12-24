# export DATA_DIR=/cfs/hadoop-aipnlp/zengweihao02/modelzoo/Llama-2-70b-hf/Llama-2-70b-hf
# #export MODEL_REPO=ScalableMath/llemma-7b-sft-metamath-level-1to3-hf

# python convert_hf_checkpoint.py \
#     --checkpoint_dir $DATA_DIR \
#     --target_precision bf16


export DATA_DIR=/cfs/hadoop-aipnlp/zengweihao02/modelzoo/Llama-3.1-8B-Instruct
#export MODEL_REPO=ScalableMath/llemma-7b-sft-metamath-level-1to3-hf


#export DATA_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/modelzoo/deepseek-math-7b-base/deepseek-math-7b-base
# python convert_hf_checkpoint.py \
#     --checkpoint_dir $DATA_DIR \
#     --target_precision bf16
#/cfs/hadoop-aipnlp/zengweihao02/b-star/easy-to-hard-main-share/scripts/convert_hf_checkpoint_llama3.py

python /cfs/hadoop-aipnlp/zengweihao02/b-star/easy-to-hard-main-share/scripts/convert_hf_checkpoint_llama3.py \
    --checkpoint_dir $DATA_DIR \
    --target_precision bf16

# python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/easy-to-hard-main-share/scripts/convert_hf_checkpoint.py \
#     --checkpoint_dir $DATA_DIR \
#     --target_precision bf16