import json

import random
import pandas as pd
import numpy as np
from collections import Counter
import re
import argparse
import os
import argparse
from itertools import product



def process_data(input_file, ref_file):
    if not os.path.exists(input_file) or not os.path.exists(ref_file):
        return []

    with open(ref_file, "r") as r:
        data_lines = r.readlines()

    infer4reward_data = [json.loads(l) for l in data_lines]

    infer_dict = {}
    for item in infer4reward_data:  # 获取官方答案
        infer_dict[item["prompt"]] = item["response"]

    with open(input_file, "r") as r:
        data_lines = r.readlines()

    data_json = []
    for item in data_lines:
        try:
            data_json.append(json.loads(item))
        except json.JSONDecodeError:
            continue

    random.shuffle(data_json)

    trn_data = {}
    for item in data_json:
        if item["idx"] in trn_data:
            trn_data[item["idx"]]["sample_list"].append({"output": item["output"], "reward": item["reward"], "response": infer_dict[item["prompt"]]})
        else:
            trn_data[item["idx"]] = {
                "prompt": item["prompt"],
                "sample_list": [{"output": item["output"], "reward": item["reward"], "response": infer_dict[item["prompt"]]}]
            }

    trn_list = []
    for item in trn_data:
        trn_list.append(trn_data[item])

    return trn_list

def get_unique_trn_json(max_samples, trn_data, score_threshold):
    trn_json = []
    for item in trn_data:
        solutions = []
        for sample in item["sample_list"]:
            if "\n\n# Answer\n\n" in sample["output"]:
                final_answer = sample["output"].split("\n\n# Answer\n\n")[-1]
                prm_score = min(sample["reward"])

                orm_score = 0.0
                if sample["output"].split("\n\n# Answer\n\n")[-1] == sample["response"].split("\n\n# Answer\n\n")[-1]:
                    orm_score = 1.0

                final_score = prm_score / 5.0 + orm_score
                solutions.append({'final_answer': final_answer, 'prm_score': prm_score, 'output': sample["output"], "score": final_score}) #计算prm_score和orm_score的加权平均, 也就是最终的reward 分数, 用于数据的筛选

        solutions.append({'final_answer': final_answer, 'prm_score': 0.0, 'output': sample["response"], "score": 1.0}) # 添加官方数据
        if len(solutions) == 0:
            continue

        solutions_sorted = sorted(solutions, key=lambda x: x['score'], reverse=True)
        idx = 0
        temp_input = item["prompt"].split("\n\n# Solution\n\n")[0]
        temp_input = temp_input.split("# Question\n\n")[-1]

        for solu in solutions_sorted:
            if solu["score"] > score_threshold:
                trn_json.append({"query": temp_input, "output": solu["output"], "response": sample["response"], "reward": solu["prm_score"]})
                idx += 1
                if idx >= max_samples:
                    break
        # 去重部分
    unique_trn_json = []
    seen = set()
    for item in trn_json:
        identifier = (item["query"], item["output"])
        if identifier not in seen:
            seen.add(identifier)
            unique_trn_json.append(item)
        
    return unique_trn_json



def cal_modi_acc_soft(data_json):
    unique_set = set()
    for item in data_json:
        unique_set.add(item["query"])
    unique_dict = {}

    for item in unique_set:
        unique_dict[item] = []

    #trn_json = []
    for item in data_json:
        unique_dict[item["query"]].append({"output": item["output"], "response": item["response"]})

    correct_num = []

    actual_num = []
    correct_ratio = []

    for item in unique_dict:
        temp_count = 0
        for output in unique_dict[item]:
            if output["output"].split("\n\n# Answer\n\n")[-1] == output["response"].split("\n\n# Answer\n\n")[-1]:
                temp_count = temp_count + 1

        correct_num.append(temp_count)
        actual_num.append(len(unique_dict[item]))
        correct_ratio.append(temp_count/len(unique_dict[item]))
    modi_acc_list = []
    for num, ratio in zip(correct_num, correct_ratio):
        if num >= 8:
            modi_acc_list.append(ratio)

        else:
            modi_acc_list.append((num * ratio)/ 8)

    return np.mean(modi_acc_list)
    


def find_rewardbase(ana_list, reward_base, target_size):
    max_samples = 1
    max_iterations = 64
    iteration = 0
    while iteration < max_iterations:
        unique_trn_json = get_unique_trn_json(max_samples, ana_list, reward_base)
        if len(unique_trn_json) >= target_size:
            break
        max_samples += 1
        iteration += 1

    modi_acc = cal_modi_acc_soft(unique_trn_json)
    random.shuffle(unique_trn_json)

    return modi_acc



def find_strategy(input, ref, target_size):
    ana_list = process_data(input, ref)  # 合并 input 和 ref, 结合数据中的 response 和 reward 数据

    if not ana_list:
        return float('-inf')
    hyper_param = -1.0

    max_modi_acc = float('-inf')

    while True:
        modi_acc = find_rewardbase(ana_list, hyper_param, target_size)
        if modi_acc > max_modi_acc:
            max_modi_acc = modi_acc

        hyper_param += 0.01

        if hyper_param > 1.0:
            break   

    return max_modi_acc


def find_dataset(input_list, ref_list, target_size):
    modi_acc_list = []


    for input_item, ref_item in zip(input_list, ref_list):
        modi_acc = find_strategy(input_item, ref_item, target_size)


        modi_acc_list.append(modi_acc)

    return modi_acc_list



def parse_args():
    parser = argparse.ArgumentParser(description="Generate input and reference file paths based on provided temps and sample numbers.")
    parser.add_argument('--temps', type=float, nargs='+', required=True, help="List of temperatures.")
    parser.add_argument('--sample_nums', type=int, nargs='+', required=True, help="List of sample numbers.")
    parser.add_argument('--input_path_template', type=str, required=True, help="Template for the input path.")
    parser.add_argument('--ref_path_template', type=str, required=True, help="Template for the reference path.")
    parser.add_argument('--iter', type=int, required=True, help="Iteration number to be included in the file path.")
    parser.add_argument('--valid_sample_size', type=int, required=True, help="Valid sample size.")
    return parser.parse_args()

def main():
    args = parse_args()

    input_list = []
    ref_list = []
    combinations = list(product(args.temps, args.sample_nums))  # 记录temp, sample_num的组合

 
    for temp, sample_num in combinations:
        # 设置输入和输出路径，将 iter 也替换到路径中
        input_path = args.input_path_template.format(temp=temp, sample_num=sample_num, iter=args.iter)
        ref_path = args.ref_path_template.format(temp=temp, sample_num=sample_num, iter=args.iter)

        input_list.append(input_path)
        ref_list.append(ref_path)

    # 假设 find_dataset 返回的是一个效果值列表，与 combinations 一一对应
    effect_values = find_dataset(input_list, ref_list, args.valid_sample_size)

    # 找到效果值最好的组合
    max_effect_index = effect_values.index(max(effect_values))
    best_temp, best_sample_num = combinations[max_effect_index]

    # 输出最好的组合，格式为：temp=最佳temp值, sample_num=最佳sample_num值
    print(f"{best_temp} {best_sample_num}")

if __name__ == "__main__":
    main()
