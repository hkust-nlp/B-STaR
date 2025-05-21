#export HF_HOME="/ssddata/model_hub"
# export CUDA_HOME=/usr/local/cuda-11.7  #指定cuda根目录
# export PATH=$PATH:/usr/local/cuda-11.7/bin  #安装的cuda的路径下的bin文件夹，包含了nvcc等二进制程序
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64  ##安装的cuda的路径下的lib64文件夹，包含很多库文件


# for i in {100..500..100}
# do
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_test.py \
#         --model /share/project/weihao/save_dir/checkpoints/math_format_infer_iter9_fix_rest_64k/iter7/converted/ckpt$i \
#         --output_path /share/project/weihao/save_dir/checkpoints/math_format_infer_iter9_fix_rest_64k/iter7/converted/ckpt$i \
#         --model_id gsm8k \
#         --start 0 \
#         --end 1400 \
#         --batch_size 800 \
#         --tensor_parallel_size 1 \
#         --max_tokens 768 
# done


# for iter in {2..6}
# do
#     for i in {100..500..100}
#     do
#         CUDA_VISIBLE_DEVICES=0 python gsm8k_test.py \
#             --model /share/project/weihao/save_dir/checkpoints/math_format_infer_iter9_fix_rewardneg0.0_pre_iterrft_64k/iter$iter/converted/ckpt$i \
#             --output_path /share/project/weihao/save_dir/checkpoints/math_format_infer_iter9_fix_rewardneg0.0_pre_iterrft_64k/iter$iter/converted/ckpt$i \
#             --model_id gsm8k \
#             --start 0 \
#             --end 1400 \
#             --batch_size 800 \
#             --tensor_parallel_size 1 \
#             --max_tokens 768
#     done
# done



# for i in {1750..2250..250}
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_test.py \
#         --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i \
#         --output_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i \
#         --model_id gsm8k \
#         --start 0 \
#         --end 1400 \
#         --batch_size 800 \
#         --tensor_parallel_size 1 \
#         --max_tokens 768 
# done

# for i in {1750..2250..250}
# do
#   CUDA_VISIBLE_DEVICES=0 python vllm_infer.py \
#     --input_data /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/easy-to-hard-main-share/data/test_ppo.json \
#     --model_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i \
#     --sample_num 1 \
#     --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
#     --tensor_parallel_size 1 \
#     --temperature 0.0 \
#     --top_k -1 \
#     --max_tokens 768
# done


# # python cal_metric_vllm.py \
# #     --tokenizer_path /share/project/weihao/model_zoo/llemma_7b/tokenizer.model \
# #     --answer_file /share/project/weihao/save_dir/checkpoints/math_format_online_rft_deepseek_math_48k_fix_bsz_upreward_update/converted/ckpt4000/test_ppo_1to5_infer_greedy.json \
# #     --output_file /share/project/weihao/save_dir/checkpoints/math_format_online_rft_deepseek_math_48k_fix_bsz_upreward_update/converted/ckpt4000/test_ppo_1to5_infer_greedy_metric.json \



# for i in {1750..2250..250}
# do
#   python cal_metric_vllm.py \
#     --tokenizer_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/modelzoo/llemma_7b/tokenizer.model \
#     --answer_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
#     --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_b_star_deepseek_math_16k_fix_bsz_fine_stand_fix_moretemp/converted/ckpt$i/test_ppo_1to5_infer_greedy_metric.json
# done


for iter in {2..10}
do
    for i in {250..250..250}
    do
        CUDA_VISIBLE_DEVICES=0 python gsm8k_test.py \
            --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i \
            --output_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i \
            --model_id gsm8k \
            --start 0 \
            --end 1400 \
            --batch_size 800 \
            --tensor_parallel_size 1 \
            --max_tokens 768
    done
done


for iter in {2..10}
do
    for i in {250..250..250}
    do
        CUDA_VISIBLE_DEVICES=0 python vllm_infer.py \
            --input_data /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/easy-to-hard-main-share/data/test_ppo.json \
            --model_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i \
            --sample_num 1 \
            --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
            --tensor_parallel_size 1 \
            --temperature 0.0 \
            --top_k -1 \
            --max_tokens 768
    done
done



for iter in {2..10}
do
    for i in {250..250..250}
    do
        python cal_metric_vllm.py \
            --tokenizer_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/modelzoo/llemma_7b/tokenizer.model \
            --answer_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
            --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_rest_llama3.1_math_32k_lr2e_5/iter$iter/converted/ckpt$i/test_ppo_1to5_infer_greedy_metric.json
    done
done


# for t in 0.5 0.7 0.9 1.1
# do
#   for i in {500..4500..500}
#   do
#       CUDA_VISIBLE_DEVICES=0 python gsm8k_test.py \
#           --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i \
#           --output_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i \
#           --model_id gsm8k \
#           --start 0 \
#           --end 1400 \
#           --batch_size 800 \
#           --tensor_parallel_size 1 \
#           --max_tokens 768 
#   done

#   for i in {500..4500..500}
#   do
#     CUDA_VISIBLE_DEVICES=0 python vllm_infer.py \
#       --input_data /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/easy2hard/share/project/weihao/easy-to-hard-main-share/easy-to-hard-main-share/data/test_ppo.json \
#       --model_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i \
#       --sample_num 1 \
#       --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
#       --tensor_parallel_size 1 \
#       --temperature 0.0 \
#       --top_k -1 \
#       --max_tokens 768
#   done

#   for i in {500..4500..500}
#   do
#     python cal_metric_vllm.py \
#       --tokenizer_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/modelzoo/llemma_7b/tokenizer.model \
#       --answer_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i/test_ppo_1to5_infer_greedy.json \
#       --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/zengweihao02/checkpoints/math_format_online_rft_mistral_math_64k_${t}t_neg0.4r/converted/ckpt$i/test_ppo_1to5_infer_greedy_metric.json
#   done
# done
