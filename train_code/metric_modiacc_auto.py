import json
import random
import pandas as pd
import numpy as np
from collections import Counter
import re
import argparse


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

def process_data(input_file, ref_file):
    with open(ref_file, "r") as r:
        data_lines = r.readlines()

    infer4reward_data = [json.loads(l) for l in data_lines]

    infer_dict = {}
    for item in infer4reward_data:
        infer_dict[item["prompt"]] = item["response"]

    with open(input_file, "r") as r:
        data_lines = r.readlines()

    data_json = []
    for item in data_lines:
        try:
            data_json.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    import random
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
            print("len(equa_list)==0")
            continue
        temp_diversity = len(equa_set) / len(equa_list)
        diversity_list.append(temp_diversity)

    return np.mean(diversity_list)


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
        if num >= 8:
            modi_acc_list.append(1.0)

        else:
            modi_acc_list.append(num / 8)

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

       

def cal_weight_acc(data_json, correct_the, ratio_the):
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

    for num, ratio in zip(correct_num, correct_ratio):
        if num >= correct_the and ratio >= ratio_the:
            acc_count = acc_count + 1

    return acc_count / len(correct_num)












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









    





def find_rewardbase(ana_list, reward_base, target_size, correct_the, ratio_the):
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
    

    #weighted_acc = cal_weight_acc(unique_trn_json, correct_the, ratio_the)


    weighted_acc = cal_modi_acc_soft(unique_trn_json)
    weighted_entropy_value = cal_weight_entropy(unique_trn_json)
    random.shuffle(unique_trn_json)

    return ui, answer_acc, mean_diversity, weighted_entropy_value, weighted_acc, unique_trn_json


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

def main(args):
    ana_list = process_data(args.input, args.ref)
    reward_list = []
    for item in ana_list:
        temp_reward = []
        for sample in item["sample_list"]:
            temp_reward.append(np.min(sample["reward"]))

        reward_list.append(np.mean(temp_reward))

    mean_reward = np.mean(reward_list)
    hyper_param = -1.0
    target_acc = 0.9  # 你可以根据需要调整 target_acc 的值



    # while True:
    #     ui, answer_acc, mean_diversity, weighted_entropy_value, unique_trn_json = find_rewardbase(ana_list, hyper_param, args.target_size)

    #     if answer_acc > args.target_acc:

    #         with open(args.output, "w") as w:
    #             json.dump(unique_trn_json, w)

    #         print("query balance", ui)
    #         print("answer acc", answer_acc)
    #         print("mean diversity", mean_diversity)
    #         break
    #     hyper_param += 0.1
    

    max_weighted_entropy = float('-inf')
    best_unique_trn_json = None
    max_weighted_acc = float('-inf')

    while True:
        ui, answer_acc, mean_diversity, weighted_entropy_value, weighted_acc, unique_trn_json = find_rewardbase(ana_list, hyper_param, args.target_size, args.correct_num, args.correct_ratio)
        # 如果当前的 weighted_entropy_value 比最大值更大，更新最大值和对应的 unique_trn_json
        if weighted_acc > max_weighted_acc:

            max_weighted_acc = weighted_acc
            max_weighted_entropy = weighted_entropy_value
            best_unique_trn_json = unique_trn_json

            best_ui = ui

            best_answer_acc = answer_acc

            best_mean_diversity = mean_diversity

            best_hyper = hyper_param

        hyper_param += 0.01

        if hyper_param > 1.0:
            break

        print("trying hyper_param...", hyper_param)
        

    # 在循环结束后保存使 weighted_entropy_value 最大的 unique_trn_json
    if best_unique_trn_json is not None:
        with open(args.output, "w") as w:
            json.dump(best_unique_trn_json, w)


    print("Max weighted Acc", max_weighted_acc)
    print("Max weighted entropy value", max_weighted_entropy)
    print("query balance", best_ui)
    print("answer acc", best_answer_acc)
    print("mean diversity", best_mean_diversity)
    print("hyper_param", best_hyper)








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process data with customizable parameters")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--ref", type=str, required=True, help="Ref file path")
    parser.add_argument("--target_size", type=int, default=1, help="Maximum number of samples per question")
    parser.add_argument("--correct_num", type=int, default=7, help="Number of Correct Number")
    parser.add_argument("--correct_ratio", type=float, default=0.9, help="Number of Correct Number")
    #parser.add_argument("--target_acc", type=float, default=0.0, help="Score threshold for filtering solutions")
    
    args = parser.parse_args()
    main(args)