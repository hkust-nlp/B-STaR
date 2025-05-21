import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from datasets import load_dataset

from datetime import datetime, date
from tqdm import tqdm
from vllm import LLM, SamplingParams
from ray_vllm import LLMRayWrapper
from string import ascii_uppercase
import re

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def construct_prompt(template, data):
    choices = data['choices']["text"]
    question = data['question'] + "\n"
    i = 0
    for choice in choices:
        question += f"({ascii_uppercase[i]}) {choice}\n"
        i += 1
    question = question.strip()
    return template.replace("{{question}}", question)

    
def extract_answer(response):
    pattern_list = [r"Final answer:\s*[\[\(]?([A-Za-z])[\]\)]?"]
    for pattern in pattern_list:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    
    if args.input_file is not None:
        problems = load_jsonl(args.input_file)
        problems = problems[args.start:args.end]
        print("Loading problems from ", min(len(problems),args.end))
        print("Number of problems: ", len(problems))
    else:
        problems = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=args.split, cache_dir=args.cache_dir)
        # problems = [{"id": idx , **item} for idx, item in enumerate(problems)]
        # random.seed(42)
        problems = problems.select(range(args.start, min(len(problems),args.end)))
    
    if not os.path.exists(os.path.dirname(args.save)):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
   

    llm = LLMRayWrapper(model_path=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=4096, cuda_ids=args.cuda_ids, swap_space = 20)  # Adjust tensor_parallel_size as needed
    stop_tokens = ["\nQUESTION:\n", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "\n\n"]
    sampling_params = SamplingParams(
        n=args.sample_num,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=stop_tokens
    )
    prompt = open(args.prompt_template_dir+"/"+str(args.shot_num)+"-shot-prompt.txt", "r").read()
    zero_shot_prompt = open(args.prompt_template_dir+"/0-shot-prompt.txt", "r").read()
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Filter too long problems with tokenizer
    # problems = problems.filter(lambda x: len(tokenizer(x["question"])["input_ids"]) < 2048)
    if not os.path.exists(os.path.dirname(args.save)):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
    # ans_file = open(args.save, "w")
   
    prompt_list = []
    for item in problems:
        prompt_list.append(construct_prompt(prompt, item))
    outputs = llm.generate(prompts = prompt_list, sampling_params = sampling_params)
    correct = 0
    gpt_codes = {}
    for i, (p, output) in enumerate(zip(problems, outputs)):
        question = p["question"]
        output_list = {
            "id": p["id"],
            "question": question,
            "input": construct_prompt(zero_shot_prompt, p),
            "gt": p["answerKey"]
        }
        for q in range(args.sample_num):
            raw_response = output.outputs[q].text
            answer = extract_answer(raw_response)
            gt = p["answerKey"]
            if answer and answer.lower() == gt.lower():
                correct += 1
                score = 1.0
            else:
                score = 0.0
            output_list["output"+str(q)] ={
                "text": raw_response,
                "score": score,
                "finish_reason": output.outputs[q].finish_reason,
                "extracted_answer": answer
            }
        gpt_codes[p["id"]] = output_list
        
        # ans_file.write(json.dumps(output_list) + "\n")
        # ans_file.flush()
        

    
        if args.debug:
            print("Prompt: ", "-" * 100)
            print(output.prompt)
            print("Completion: ", "-" * 100)
            print(output_list['output0']['text'])
            print("Ground Truth: ", gt)
            print("Score: ", output_list['output0']['score'])
    # ans_file.close()
    
    with open(args.save, "w") as f:
        json.dump(gpt_codes, f, indent=2)

    print(f"Accuracy: {correct / (len(problems) * args.sample_num)}")
    
    
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate ARC Challenge.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt_template_dir", default=None, type=str)
    parser.add_argument("--shot_num", default=None, type=int)
    parser.add_argument("--input_file", default=None, type=str)
    # parser.add_argument("-t","--test_loc", default="~/apps/data_split/test.json", type=str, help="path to the test folder.")
    # parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    # parser.add_argument("-l","--load", default="", type=str)
    # parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--sample_num", type=int, default=10) # 采样数量
    parser.add_argument("--temperature", type=float, default=1.0) # 温度
    parser.add_argument("--top_k", type=int, default=20) # top_k
    parser.add_argument("--max_tokens", type=int, default=768) # 最大token数
    # parser.add_argument("--difficulty", default="introductory", type=str)
    # parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=10000000, type=int)
    # parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("--cuda_ids", type=str, default="0")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--split", type=str, default="train", help="What split to use.")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vllm")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    args = parser.parse_args()

    main(args)
