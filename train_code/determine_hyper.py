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

# def process_data(input_file, ref_file):
#     with open(ref_file, "r") as r:
#         data_lines = r.readlines()

#     infer4reward_data = [json.loads(l) for l in data_lines]

#     infer_dict = {}
#     for item in infer4reward_data: #获取官方答案
#         infer_dict[item["prompt"]] = item["response"]

#     with open(input_file, "r") as r:
#         data_lines = r.readlines()

#     data_json = []
#     for item in data_lines:
#         try:
#             data_json.append(json.loads(item))
#         except json.JSONDecodeError:
#             continue
#     import random
#     random.shuffle(data_json)

#     trn_data = {}
#     for item in data_json:
#         if item["idx"] in trn_data:
#             trn_data[item["idx"]]["sample_list"].append({"output": item["output"], "reward": item["reward"], "response": infer_dict[item["prompt"]]})
#         else:
#             trn_data[item["idx"]] = {
#                 "prompt": item["prompt"],
#                 "sample_list": [{"output": item["output"], "reward": item["reward"], "response": infer_dict[item["prompt"]]}]
#             }

#     trn_list = []


#     for item in trn_data:
#         trn_list.append(trn_data[item]) 

#     return trn_list


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
                solutions.append({'final_answer': final_answer, 'prm_score': prm_score, 'output': sample["output"], "score": final_score})

        solutions.append({'final_answer': final_answer, 'prm_score': 0.0, 'output': sample["response"], "score": 1.0})
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


def uniformity_index(data):
    # 统计每个唯一样本的频次
    freq_counts = Counter(data)
    
    # 总样本数 N
    N = len(data)
    # 唯一样本数 M
    M = len(freq_counts)
    
    # 期望频次
    expected_freq = N / M
    
    # 实际频次列表
    actual_freqs = np.array(list(freq_counts.values()))
    
    # 计算标准差
    std_dev = np.sqrt(np.mean((actual_freqs - expected_freq) ** 2))
    
    # 计算 Uniformity Index
    uniformity_index = 1 - (std_dev / expected_freq)
    
    return uniformity_index


def cal_acc(data):
    count_acc = 0.0

    for item in data:
        if item["output"].split("\n\n# Answer\n\n")[-1] == item["response"].split("\n\n# Answer\n\n")[-1]:
            count_acc = count_acc + 1


    return count_acc / len(data)


def cal_diversity(data):
    data_dict = {}
    data_set = set()
    diversity_list = []
    for item in data:
        data_set.add(item["query"])

    for item in data_set:
        data_dict[item] = []

    for item in data:
        data_dict[item["query"]].append(item["output"])

    for item in data_dict:
        equa_list = []
        equa_set = set()

        for i in data_dict[item]:
            temp_equa = re.findall(r'\$(.*?)\$|<<([^<>]*)>>|\\\[(.*?)\\\]', i, re.DOTALL)
            # 将匹配到的结果平铺为一个列表
            temp_equa = [eq for match in temp_equa for eq in match if eq]

            for e in temp_equa:
                equa_list.append(e)
                equa_set.add(e)
        if len(equa_list) == 0:
            #print("len(equa_list)==0")
            continue
        temp_diversity = len(equa_set) / len(equa_list)
        diversity_list.append(temp_diversity)

    return np.mean(diversity_list)


def cal_weight_entropy(data_json):
    unique_set = set()
    for item in data_json:
        unique_set.add(item["query"])
    unique_dict = {}

    for item in unique_set:
        unique_dict[item] = []

    trn_json = []
    for item in data_json:
        unique_dict[item["query"]].append({"output": item["output"], "response": item["response"]})

    frequencies_list = []
    weights_list = []

    max_sample = 0.0
    for item in unique_dict:
        temp_freq = len(unique_dict[item]) / len(data_json)

        if len(unique_dict[item]) > max_sample:
            max_sample = len(unique_dict[item])

        temp_count = 0
        for output in unique_dict[item]:
            if output["output"].split("\n\n# Answer\n\n")[-1] == output["response"].split("\n\n# Answer\n\n")[-1]:
                temp_count = temp_count + 1
        frequencies_list.append(temp_freq)
        temp_acc = temp_count
        weights_list.append(temp_acc) 
    

    weights_list = np.array(weights_list) / max_sample
    weighted_entropy_value = weighted_entropy(frequencies_list, weights_list)

    return weighted_entropy_value






def cal_modi_acc(data_json):
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
    acc_count = 0.0

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
        if num >= 7:
            modi_acc_list.append(1.0)

        else:
            modi_acc_list.append(num / 7)

    return np.mean(modi_acc_list)  

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
    acc_count = 0.0

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
    
def cal_modi_acc_fix(data_json):
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
    acc_count = 0.0

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
            modi_acc_list.append(num / 8)

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
    #unique_trn_json = unique_trn_json[:target_size]
    unique_list = [item["query"] for item in unique_trn_json]
    ui = uniformity_index(unique_list)
    answer_acc = cal_acc(unique_trn_json)
    mean_diversity = cal_diversity(unique_trn_json)

    weighted_entropy_value = cal_weight_entropy(unique_trn_json)
    modi_acc = cal_modi_acc_soft(unique_trn_json)
    random.shuffle(unique_trn_json)

    return ui, answer_acc, mean_diversity, weighted_entropy_value, unique_trn_json, modi_acc


def weighted_entropy(frequencies, weights):
    """
    Calculate the weighted entropy based on sample frequencies and weights.
    
    Parameters:
    frequencies (list or np.array): Array of sample frequencies.
    weights (list or np.array): Array of corresponding weights.
    
    Returns:
    float: Weighted entropy.
    """
    # Normalize frequencies to get probabilities
    probabilities = frequencies / np.sum(frequencies)
    
    # Calculate weighted entropy
    weighted_entropy = -np.sum(weights * probabilities * np.log(probabilities))
    
    return weighted_entropy




def find_strategy(input, ref, target_size):
    ana_list = process_data(input, ref)

    if not ana_list:
        return [], float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')
    reward_list = []
    for item in ana_list:
        temp_reward = []
        for sample in item["sample_list"]:
            temp_reward.append(np.min(sample["reward"]))

        reward_list.append(np.mean(temp_reward))

    mean_reward = np.mean(reward_list)
    hyper_param = -1.0
    target_acc = 0.9  # 你可以根据需要调整 target_acc 的值


    max_weighted_entropy = float('-inf')
    best_unique_trn_json = None
    max_modi_acc = float('-inf')

    while True:
        ui, answer_acc, mean_diversity, weighted_entropy_value, unique_trn_json, modi_acc = find_rewardbase(ana_list, hyper_param, target_size)
        # 如果当前的 weighted_entropy_value 比最大值更大，更新最大值和对应的 unique_trn_json
        #if weighted_entropy_value > max_weighted_entropy:
        if modi_acc > max_modi_acc:
            max_weighted_entropy = weighted_entropy_value
            best_unique_trn_json = unique_trn_json
            max_modi_acc = modi_acc
            best_ui = ui

            best_answer_acc = answer_acc

            best_mean_diversity = mean_diversity

            best_hyper = hyper_param

        hyper_param += 0.01

        if hyper_param > 1.0:
            break   

    return best_unique_trn_json, max_weighted_entropy, best_ui, best_answer_acc, best_mean_diversity, best_hyper, max_modi_acc


def find_dataset(input_list, ref_list, target_size):
    best_unique_trn_json_list, max_weighted_entropy_list, best_ui_list, best_answer_acc_list, best_mean_diversity_list, best_hyper_list, modi_acc_list = [], [], [], [], [], [], []


    for input_item, ref_item in zip(input_list, ref_list):
        best_unique_trn_json, max_weighted_entropy, best_ui, best_answer_acc, best_mean_diversity, best_hyper, modi_acc = find_strategy(input_item, ref_item, target_size)

        best_hyper_list.append(best_hyper)
        best_mean_diversity_list.append(best_mean_diversity)
        best_unique_trn_json_list.append(best_unique_trn_json)

        max_weighted_entropy_list.append(max_weighted_entropy)

        best_ui_list.append(best_ui)
        best_answer_acc_list.append(best_answer_acc)

        modi_acc_list.append(modi_acc)

    return modi_acc_list

    #print("bupt")


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

    # print("args.temps",args.temps)
    #print("combinations", combinations)



    for temp, sample_num in combinations:
        # 设置输入和输出路径，将 iter 也替换到路径中
        input_path = args.input_path_template.format(temp=temp, sample_num=sample_num, iter=args.iter)
        ref_path = args.ref_path_template.format(temp=temp, sample_num=sample_num, iter=args.iter)

        input_list.append(input_path)
        ref_list.append(ref_path)

        #import pdb
        #pdb.set_trace()

    # 假设 find_dataset 返回的是一个效果值列表，与 combinations 一一对应
    effect_values = find_dataset(input_list, ref_list, args.valid_sample_size)

    # 找到效果值最好的组合
    max_effect_index = effect_values.index(max(effect_values))
    best_temp, best_sample_num = combinations[max_effect_index]

    # 输出最好的组合，格式为：temp=最佳temp值, sample_num=最佳sample_num值
    print(f"{best_temp} {best_sample_num}")

if __name__ == "__main__":
    main()