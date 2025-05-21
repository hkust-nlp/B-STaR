# 基于diversity筛选出的数据，进行self-consistency

from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random

import re
import argparse

def sample_resp(args):
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
    stop_tokens = ["\nQUESTION:\n", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "# your code here", "QUESTION", "# Your code goes here", "# Write your code", "\n\n\n\n", "<|end_of_text|>", "\n\nSolved"]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,  n=args.sample_num, stop=stop_tokens)
    all_input = [] 

    for item in sample_data:
        #user_input = prompt_template.replace("{instruction}", item)
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)  # start index
    parser.add_argument("--model_dir", type=str)  # start index
    parser.add_argument("--sample_num", type=int, default=10) #start index
    parser.add_argument("--temperature", type=float, default=1.0) #start index
    parser.add_argument("--top_k", type=int, default=20) #start index
    parser.add_argument("--max_tokens", type=int, default=768) #start index
    parser.add_argument("--output_file", type=str)  # start index
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    sample_resp(args=args)

