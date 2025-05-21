import argparse
import json
import pdb
import re

import jsonlines
from fraction import Fraction

from vllm import LLM, SamplingParams
#from extract_answer_use_chatgpt import _extract_answer_chatgpt
import sys

MAX_INT = sys.maxsize #最大整数

####CUDA_VISIBLE_DEVICES=0 python gsm8k_test_the_answer_is_original_batch.py --model xxxxx --start 0 --end 1400 --batch_size 80 --tensor_parallel_size 1

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('\n\n# Answer\n\n')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:  ######gsm8k答案都是整数
            if '/' in match.group():
                denominator = match.group().split('/')[1]  # 分母
                numerator = match.group().split('/')[0]  # 分子
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':  ##分母为0

                        print('分母为0 ====:', match.group())
                        return round(float(numerator.replace(',', '')))
                    else:  ##分母不为0
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))  # 分数， 四舍五入取整
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))  ###小数和千分数, 四舍五入取整
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(args, model, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []


    #problem_prompt = ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT: Let's think step by step.")
    #problem_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

    problem_prompt = "# Question\n\n{instruction}\n\n# Solution\n\n"
    #problem_prompt = ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant's response is from ChatGPT3.5. USER: {instruction} ASSISTANT: Let's think step by step.")
    # problem_prompt = (
    #     "You are a math assistant, skilled at solving various mathematical problems. USER: {instruction} ASSISTANT: Let's think step by step.")
    print('promt =====', problem_prompt)
    with open('./test.jsonl',"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)
    
    


    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=args.max_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        print('llm idx ====', idx)
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    all_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        print('chatgpt idx =====', idx)
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            all_outputs.append(temp)
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
            all_outputs.append(temp)
        # pdb.set_trace()
    acc = sum(result) / len(result)
    print('start===', start, ', end====', end)
    print('length====', len(result), ', acc====', acc)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    model_name = '_'.join(model.split('/')[-2:])
    
    OUTPUT_FILE_SUFFIX = ".json"
    import os
    output_data_path =  os.path.join(args.output_path, args.model_id + OUTPUT_FILE_SUFFIX)
    import os
    output_dir = os.path.dirname(output_data_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    with open(output_data_path, 'w') as f:
        json.dump({"length====": len(result), "acc====": acc}, f)

        
    #pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # start index
    parser.add_argument("--model_id", type=str)  # start index
    parser.add_argument("--output_path", type=str)  # start index
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # start index
    parser.add_argument("--batch_size", type=int, default=1)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--max_tokens", type=int, default=768)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(args=args, model=args.model, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size) 
    #pdb.set_trace()
