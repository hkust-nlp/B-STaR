from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import re
import argparse
# import wandb
# from huggingface_hub import HfApi
# from huggingface_hub import HfApi, upload_file

# def upload_to_huggingface(output_file, repo_id, token, commit_message="Upload output file"):
#     # Initialize HfApi with token
#     api = HfApi()
    
#     # Upload the file
#     api.upload_file(
#         path_or_fileobj=output_file,
#         path_in_repo=output_file,  # You can change the path in the repo if needed
#         repo_id=repo_id,
#         token=token,
#         commit_message=commit_message
#     )

def upload_to_huggingface(output_file, repo_id, token, repo_type='dataset', commit_message="Upload output file"):
    # Upload the file to the specified repository
    upload_file(
        path_or_fileobj=output_file,
        path_in_repo=output_file,  # You can change the path in the repo if needed
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
        commit_message=commit_message
    )
    print(f"File {output_file} uploaded successfully to {repo_id}.")

def sample_resp(args):
    # wandb.init(project=args.wandb_project, config={
    #     "input_data": args.input_data,
    #     "model_dir": args.model_dir,
    #     "sample_num": args.sample_num,
    #     "temperature": args.temperature,
    #     "top_k": args.top_k,
    #     "max_tokens": args.max_tokens,
    #     "output_file": args.output_file,
    #     "tensor_parallel_size": args.tensor_parallel_size,
    # })

    with open(args.input_data, "r") as r:
        data_json = json.load(r)
    sample_data = data_json

    data_dict = {}

    for item in sample_data:
        data_dict[item["input"]] = item

    prompt_template = '''

    "Below is an instruction that describes a task. " "Write a response that appropriately completes the request.\n\n" "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

    '''
    llm = LLM(model=args.model_dir, tensor_parallel_size=args.tensor_parallel_size)

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,  n=args.sample_num)
    all_input = []

    for item in sample_data:
        all_input.append(item["input"])
    outputs = llm.generate(all_input, sampling_params)

    all_output = []
    all_prompt = []

    all_json = []

    for output in outputs:

        temp_json = data_dict[output.prompt]
        all_prompt.append(output.prompt)

        temp_json["prompt"] = output.prompt

        for i in range(args.sample_num):
            temp_json["output"+str(i)] = output.outputs[i].text

        all_json.append(temp_json)

    with open(args.output_file, "w") as w:
        json.dump(all_json, w)

    # 上传文件到Hugging Face
    #upload_to_huggingface(args.output_file, args.repo_id, args.hf_token)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)  # 输入数据文件路径
    parser.add_argument("--model_dir", type=str, required=True)  # 模型目录
    parser.add_argument("--sample_num", type=int, default=10) # 采样数量
    parser.add_argument("--temperature", type=float, default=1.0) # 温度
    parser.add_argument("--top_k", type=int, default=20) # top_k
    parser.add_argument("--max_tokens", type=int, default=768) # 最大token数
    parser.add_argument("--output_file", type=str, required=True)  # 输出文件路径
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # 并行大小
    #parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")  # Hugging Face存储库ID
    #parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")  # Hugging Face访问令牌
    #parser.add_argument("--wandb_project", type=str, default="my_project")  # wandb project name
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample_resp(args=args)
