import json
from models.tokenizer_utils import AcceleraTokenizer
#from math_utils.math_rl_utils import post_process_math_rollouts

from typing import Dict, List, Optional, Tuple
import torch
import trainers.common_utils as common_utils
from itertools import chain
from math_utils import grader
import torch.distributed as dist
from pathlib import Path
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None
def _calculate_outcome_accuracy(
    predicted_answers: List[str],
    gt_answers: List[str],
    answers: List[str],
    levels: List[int],
    outcome_reward: bool,
    easy_outcome_reward: bool,
    device: torch.device,
):
    assert len(predicted_answers) == len(answers)

    assert not (
        outcome_reward and easy_outcome_reward
    ), "Cannot use both outcome_reward and easy_outcome_reward."

    with common_utils.DisableLogger():
        outcome_accuracy = [
            1.0 if grader.grade_answer(predicted_answer, gt_answer) else 0.0
            for predicted_answer, gt_answer in zip(predicted_answers, gt_answers)
        ]

        # TODO (zhiqings): 0.25 is a magic number.
        unavailable_reward = 0.25
        if outcome_reward:
            symbolic_rewards = outcome_accuracy
        elif easy_outcome_reward:
            symbolic_rewards = []
            for predicted_answer, answer in zip(predicted_answers, answers):
                if answer == "Unavailable":
                    score = unavailable_reward
                elif grader.grade_answer(predicted_answer, answer):
                    score = 1.0
                else:
                    score = 0.0
                symbolic_rewards.append(score)
        else:
            symbolic_rewards = [
                unavailable_reward for _ in range(len(predicted_answers))
            ]

    assert len(symbolic_rewards) == len(predicted_answers)

    per_level_counts = {}
    per_level_accuracy = {}

    all_unique_levels = list(range(1, 6))

    for level in all_unique_levels:
        per_level_counts[level] = []
        per_level_accuracy[level] = []

    for level, accuracy in zip(levels, outcome_accuracy):
        for unique_level in all_unique_levels:
            if level == unique_level:
                per_level_counts[unique_level].append(1.0)
                per_level_accuracy[unique_level].append(accuracy)
            else:
                per_level_counts[unique_level].append(0.0)
                per_level_accuracy[unique_level].append(0.0)

    for level in all_unique_levels:
        assert len(per_level_counts[level]) == len(outcome_accuracy)
        assert len(per_level_accuracy[level]) == len(outcome_accuracy)
        per_level_counts[level] = torch.tensor(per_level_counts[level], device=device)
        per_level_accuracy[level] = torch.tensor(
            per_level_accuracy[level], device=device
        )

    original_symbolic_rewards = symbolic_rewards

    symbolic_rewards = torch.tensor(symbolic_rewards, device=device)
    outcome_accuracy = torch.tensor(outcome_accuracy, device=device)

    ret_dict = {
        "symbolic_rewards": symbolic_rewards,
        "outcome_accuracy": outcome_accuracy,
    }

    for level in sorted(list(all_unique_levels)):
        ret_dict[f"level_{level}_counts"] = per_level_counts[level]
        ret_dict[f"level_{level}_accuracy"] = per_level_accuracy[level]

    return ret_dict, original_symbolic_rewards
def merge_fn(tensor_or_list):
    if isinstance(tensor_or_list[0], list):
        return list(chain(*tensor_or_list))
    else:
        return torch.cat(tensor_or_list, dim=0)
def post_process_math_rollouts(
    text_responses: List[str],
    answers: List[str],
    gt_answers: List[str],
    levels: List[str],
    tokenizer: AcceleraTokenizer,
    stop_token: Optional[str],
    outcome_reward: bool,
    easy_outcome_reward: bool,
    device: torch.device,
):
    if stop_token is not None:
        parsed_stop_token = stop_token
        parsed_stop_token = parsed_stop_token.replace(r"\n", "\n")
        parsed_stop_token = parsed_stop_token.replace(r"\\", "\\")
    else:
        parsed_stop_token = tokenizer.eos_token

    predicted_answers = []
    for text_response in text_responses:
        predicted_answer = "No answer found."
        if "\n\n" in parsed_stop_token:
            if parsed_stop_token in text_response:
                predicted_answer = text_response.split(parsed_stop_token)[1]
                predicted_answer = predicted_answer.split(tokenizer.eos_token)[0]
        elif "\\boxed{}" == parsed_stop_token:
            boxed_predicted_answer = text_response.split(tokenizer.eos_token)[0]
            boxed_predicted_answer = remove_boxed(
                last_boxed_only_string(boxed_predicted_answer)
            )
            if boxed_predicted_answer is not None:
                predicted_answer = boxed_predicted_answer
        else:
            raise ValueError(f"Unknown stop token: {parsed_stop_token}")
        predicted_answers.append(predicted_answer)

    # text_answers_gt_levels = tokenizer.batch_decode(
    #     answer_gt_levels,
    #     skip_special_tokens=True,
    # )

    #answers, gt_answers, levels = [], [], []
    # for text_answers_gt_level in text_answers_gt_levels:
    #     assert len(text_answers_gt_level.split(";;;")) == 3, text_answers_gt_level
    #     answer, gt_answer, level = text_answers_gt_level.split(";;;")
    #     answers.append(answer.strip())
    #     gt_answers.append(gt_answer.strip())
    #     levels.append(int(level.strip()))

    outcome_metrics, symbolic_rewards = _calculate_outcome_accuracy(
        predicted_answers,
        gt_answers,
        answers,
        levels,
        outcome_reward,
        easy_outcome_reward,
        device,
    )
    return (
        predicted_answers,
        gt_answers,
        levels,
        symbolic_rewards,
        outcome_metrics,
    )
def main(
    tokenizer_path: Path = Path(
        "/ssddata/weihao00/model_zoo/llemma_7b/checkpoints/EleutherAI/llemma_7b/tokenizer.model"
    ),

    answer_file: Path = Path(
        "/ssddata/weihao00/easy2hard/easy-to-hard-main/data/test_ppo.json"    
    ),
    output_file: Path = Path(
        "/ssddata/weihao00/easy2hard/save_file/test_ppo_infer_metric.json"  
    ),



    ):

    tokenizer = AcceleraTokenizer(tokenizer_path)
    tokenizer.pad_id = tokenizer.unk_id

    with open(answer_file, "r") as r:
        test_ppo = json.load(r)


    # with open(prompt_file , "r") as r:
    #     data_lines = r.readlines()

    # data_json = [json.loads(l) for l in data_lines]
    #test_ppo = data_json
    # for idx, item in enumerate(test_ppo):
    #     item["idx"] = idx

    #     for i in data_json:
    #         if i["idx"] == idx:
    #             item["prompt"] = i["prompt"]

    #             item["output"] = i["output"]


    test_queries = []
    test_responses = []
    answers_list = []
    gt_answers_list = []
    levels_list = []
    for item in test_ppo:
        test_queries.append(item["input"])
        test_responses.append(item["output0"])
        answers_list.append(item["answer"])
        gt_answers_list.append(item["gt_answer"])
        levels_list.append(item["level"])

    eval_rollouts_batch = {}
    eval_rollouts_batch["text_queries"] = test_queries
    eval_rollouts_batch["text_responses"] = test_responses

    outcome_metrics = post_process_math_rollouts(test_responses, answers_list, gt_answers_list, levels_list, tokenizer, "\n\n# Answer\n\n", False, False, torch.device('cpu'))

    eval_rollouts_batch.update(outcome_metrics[-1])
    cpu_eval_rollouts = []

    cpu_eval_rollouts.append(
                {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in eval_rollouts_batch.items()
                }
            )
    eval_rollouts = cpu_eval_rollouts

    eval_rollouts = common_utils.merge_dict(eval_rollouts, merge_fn=merge_fn)

    eval_stats = {}
    overall_counts = 0.0
    overall_accuracy = 0.0
    for level in range(9):
        if f"level_{level}_counts" in eval_rollouts:
            level_counts = eval_rollouts[f"level_{level}_counts"].sum()
            level_accuracy = eval_rollouts[f"level_{level}_accuracy"].sum()
            overall_counts += level_counts
            overall_accuracy += level_accuracy

            eval_stats[f"accuracy_level_{level}"] = level_accuracy / level_counts

            eval_stats[f"counts_level_{level}"] = level_counts

    eval_stats[f"accuracy_overall"] = overall_accuracy.view(1) / (
            overall_counts.view(1)
        )
    eval_stats[f"counts_overall"] = overall_counts
    eval_stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in eval_stats.items()
        }
    print(eval_stats)
    with open(output_file, "w") as w:
        json.dump(eval_stats, w)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--answer_file",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File to write generated samples to.",
    )
    args = parser.parse_args()
    main(
        args.tokenizer_path,
        args.answer_file,
        args.output_file,

    )


# tokenizer = AcceleraTokenizer("/ssddata/weihao00/model_zoo/llemma_7b/checkpoints/EleutherAI/llemma_7b/tokenizer.model")
# tokenizer.pad_id = tokenizer.unk_id
# with open("/ssddata/weihao00/easy2hard/easy-to-hard-main/data/test_ppo.json", "r") as r:
#     test_ppo = json.load(r)

# with open("/ssddata/weihao00/easy2hard/save_file/test_ppo_infer.json" , "r") as r:
#     data_lines = r.readlines()


# data_set = set()

# data_json = [json.loads(l) for l in data_lines]

# # for item in data_json:
# #     data_set.add(item)


# for idx, item in enumerate(test_ppo):
#     item["idx"] = idx

#     for i in data_json:
#         if i["idx"] == idx:
#             item["prompt"] = i["prompt"]

#             item["output"] = i["output"]


# test_queries = []
# test_responses = []
# answers_list = []
# gt_answers_list = []
# levels_list = []
# for item in test_ppo:
#     test_queries.append(item["input"])
#     test_responses.append(item["output"])
#     answers_list.append(item["answer"])
#     gt_answers_list.append(item["gt_answer"])
#     levels_list.append(item["level"])
# eval_rollouts_batch = {}
# eval_rollouts_batch["text_queries"] = test_queries
# eval_rollouts_batch["text_responses"] = test_responses

# outcome_metrics = post_process_math_rollouts(test_responses, answers_list, gt_answers_list, levels_list, tokenizer, "\n\n# Answer\n\n", False, False, "cuda:6")

# eval_rollouts_batch.update(outcome_metrics[-1])
# cpu_eval_rollouts = []

# cpu_eval_rollouts.append(
#                 {
#                     key: value.cpu() if torch.is_tensor(value) else value
#                     for key, value in eval_rollouts_batch.items()
#                 }
#             )
# eval_rollouts = cpu_eval_rollouts

# eval_rollouts = common_utils.merge_dict(eval_rollouts, merge_fn=merge_fn)
# # filtered_eval_rollouts = {}

# # for key, value in eval_rollouts.items():
# #     filtered_eval_rollouts[key] = value[:eval_data_size]
# # eval_rollouts = filtered_eval_rollouts

# eval_stats = {}
# overall_counts = 0.0
# overall_accuracy = 0.0
# for level in range(9):
#     if f"level_{level}_counts" in eval_rollouts:
#         level_counts = eval_rollouts[f"level_{level}_counts"].sum()
#         level_accuracy = eval_rollouts[f"level_{level}_accuracy"].sum()
#         overall_counts += level_counts
#         overall_accuracy += level_accuracy

#         eval_stats[f"accuracy_level_{level}"] = level_accuracy / level_counts

#         eval_stats[f"counts_level_{level}"] = level_counts

# eval_stats[f"accuracy_overall"] = overall_accuracy.view(1) / (
#             overall_counts.view(1)
#         )
# eval_stats[f"counts_overall"] = overall_counts
# eval_stats = {
#             key: value.item() if torch.is_tensor(value) else value
#             for key, value in eval_stats.items()
#         }
# print("bupt")













# print("bupt")