# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict
from collections import OrderedDict
import itertools
import fcntl

import torch

import torch._inductor.config
import torch._dynamo.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.reward_model import RewardModel, apply_reward_modeling_head
from models.tp import (
    maybe_init_dist,
    initialize_model_parallel,
    apply_tp,
    apply_reward_head_tp,
    get_model_parallel_rank,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
# from data_utils.tokenizer_utils import (
#     FakePreTrainedTokenizer,
#     batch_encode_tokens,
# )

from models.tokenizer_utils import (
AcceleraTokenizer, 
batch_encode_tokens,
)
import re
from typing import Dict, Optional, Union, Any, List, Tuple
def batch_encode_tokens_llama3(
    tokenizer: AcceleraTokenizer,
    strings: List[str],
    bos=True,
    eos=False,
    device="cuda",
    padding_side="left",
):
    assert padding_side in ["left", "right"]
    batched_tokens = []
    max_len = 0

    for string in strings:
        # Split the string on "\n\n", including the delimiters
        segments = re.split('(\n\n)', string)
        tokens = []
        is_first_token = True

        for segment in segments:
            if segment == '\n\n':

                # using reserved token id
                tokens.extend([128011,128011])
            else:
                # Encode the segment
                segment_tokens = tokenizer.encode(segment, bos=False, eos=False)
                # Add BOS token only at the very beginning
                if is_first_token and bos:
                    segment_tokens = [tokenizer.bos_id] + segment_tokens
                    is_first_token = False
                tokens.extend(segment_tokens)

        # Add EOS token at the end if specified
        if eos:
            tokens.append(tokenizer.eos_id)

        batched_tokens.append(tokens)
        max_len = max(max_len, len(tokens))

    # Determine pad_id
    pad_id = tokenizer.pad_id if tokenizer.pad_id >= 0 else tokenizer.unk_id

    # Prepare for padding
    if padding_side == "left":
        left_pad_mask_pos = torch.zeros(
            (len(batched_tokens),), dtype=torch.int, device=device
        )
    is_padded = False

    # Pad the token sequences
    for i in range(len(batched_tokens)):
        if len(batched_tokens[i]) < max_len:
            pad_len = max_len - len(batched_tokens[i])

            if padding_side == "left":
                batched_tokens[i] = [pad_id] * pad_len + batched_tokens[i]
                left_pad_mask_pos[i] = pad_len
            else:
                batched_tokens[i] = batched_tokens[i] + [pad_id] * pad_len

            is_padded = True

    # Return the appropriate output based on padding side
    if padding_side == "left":
        if not is_padded:
            left_pad_mask_pos = None

        return (
            torch.tensor(batched_tokens, dtype=torch.int, device=device),
            left_pad_mask_pos,
        )
    else:
        return torch.tensor(batched_tokens, dtype=torch.int, device=device)


# from training_utils.checkpoint_hook import (
#     get_latest_checkpoint_path,
#     load_inference_checkpoint,
# )

from checkpoint_utils import (
    get_latest_checkpoint_path,
    load_inference_checkpoint,
    load_inference_checkpoint_mistral,
)


def model_forward(model, x):
    return model(x)


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}

    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()

    return all_backward_hooks


@torch.no_grad()
def model_score(
    model: RewardModel,
    prompt: torch.Tensor,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Scores a batch of prompts using a reward model.
    """
    B, T = prompt.size(0), prompt.size(1)

    max_seq_len = max_seq_len or T

    device = prompt.device
    with torch.device(device):
        model.backbone_model.setup_caches(
            max_batch_size=B, max_seq_length=max_seq_len, kv_cache=False
        )

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_mem_efficient=False, enable_math=False
    ):
        rewards = model(prompt)

    return rewards


def _load_reward_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = RewardModel.from_name(checkpoint_path.parent.name)

        print(model)

        print(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for reward model!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for reward model!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.backbone_model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model.backbone_model)

    #apply_reward_modeling_head(model.backbone_model)

    if use_tp:
        print("Applying tensor parallel to reward head ...")
        apply_reward_head_tp(model.backbone_model)

    model = model.to(device=device, dtype=precision)
    return model.eval()



def _load_reward_model_llama3(checkpoint_path, device, precision, use_tp):
    #with torch.device("meta"):
        #model = RewardModel.from_name(checkpoint_path.parent.name)
    model = RewardModel.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=False,
            freeze_norm=False,
            freeze_output=False,
    )
    print(model)

    print(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for reward model!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for reward model!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)

    # 获取模型的 state_dict
    model_state_dict = model.backbone_model.state_dict()

    # 创建一个新的字典，只包含与模型匹配的权重
    filtered_checkpoint = {}
    for k, v in checkpoint.items():
        if k in model_state_dict and model_state_dict[k].size() == v.size():
            filtered_checkpoint[k] = v
        
        else:
            print(f"Skipping {k} due to size mismatch.")

    # 先将模型移到实际设备再加载权重
    #model = model.to(device=device, dtype=precision)
    # 加载过滤后的 checkpoint
    model.backbone_model.load_state_dict(filtered_checkpoint, assign=True, strict=False)

    #model.backbone_model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model.backbone_model)

    #apply_reward_modeling_head(model.backbone_model)

    if use_tp:
        print("Applying tensor parallel to reward head ...")
        apply_reward_head_tp(model.backbone_model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def main(
    prompt_file: Path,
    output_file: Path,
    batch_size: int = 4,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    finetune_checkpoint_path: Optional[Path] = None,
    resume_generation: bool = False,
    tensor_parallel_size: Optional[int] = None,
    on_the_fly_8bit_quantization: bool = False,
    process_reward_with_answer: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"


    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent
    #assert tokenizer_path.is_file(), tokenizer_path

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    tp_size = 1
    if use_tp:
        tp_size = tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()

    # if "Meta-Llama-3" in str(checkpoint_path):
    #     model = _load_reward_model_llama3(checkpoint_path, device, precision, use_tp)
    # else:
    model = _load_reward_model(checkpoint_path, device, precision, use_tp)
    #print(model.backbone_model.output.weight)
    #print(model)
    if finetune_checkpoint_path is not None:
        finetune_checkpoint_path, _, _ = get_latest_checkpoint_path(
            finetune_checkpoint_path
        )

        print("finetune_checkpoint_path", finetune_checkpoint_path)

        print("Loading finetune model ...")

        if finetune_checkpoint_path is not None:
            if "Mistral-7B-v0.1" in checkpoint_path.parent.name:
                load_inference_checkpoint_mistral(finetune_checkpoint_path, model)

            elif "Meta-Llama-3" in checkpoint_path.parent.name:
                load_inference_checkpoint_mistral(finetune_checkpoint_path, model)

            elif "Llama-3.1" in checkpoint_path.parent.name:
                load_inference_checkpoint_mistral(finetune_checkpoint_path, model)
            else:

                load_inference_checkpoint(finetune_checkpoint_path, model)

        model = model.eval()

        print(model.backbone_model.output)
        print(model.backbone_model.output.weight)

    if on_the_fly_8bit_quantization:
        print("Quantizing model ...")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime_on_the_fly()
        model = model.to(device=device)
        model = model.eval()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AcceleraTokenizer(tokenizer=tokenizer_path)

    torch.manual_seed(1234)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )
    if compile:
        global model_forward
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", fullgraph=True
        )

    prompts = []

    with open(prompt_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    # sort prompts by length to minimize padding

    # # debug
    # prompts = prompts[:1000]

    assert "idx" in prompts[0]
    assert "sample_idx" in prompts[0]

    all_full_seq = [prompt["prompt"] + prompt["output"] for prompt in prompts]

    print("Tokenizing prompts ...")
    tokenized_full_seq = tokenizer.batch_encode(
        all_full_seq, bos=[False] * len(all_full_seq), eos=[False] * len(all_full_seq)
    )

    #print("tokenized_full_seq", tokenized_full_seq)

    for prompt, tokenized in zip(prompts, tokenized_full_seq):
        prompt["full_seq"] = prompt["prompt"] + prompt["output"]
        prompt["full_seq_len"] = len(tokenized)

    prompts = sorted(prompts, key=lambda x: x["full_seq_len"])

    skipped_prompt_sample_ids = set()

    if rank == 0 or not use_tp:
        output_parent = output_file.parent
        if not output_parent.is_dir():
            output_parent.mkdir(exist_ok=True, parents=True)

    if use_tp:
        torch.distributed.barrier()

    print("Skipping prompts that have already been generated ...")
    if resume_generation and os.path.isfile(output_file):
        with open(output_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                prompt_sample_ids = (sample["idx"], sample["sample_idx"])
                skipped_prompt_sample_ids.add(prompt_sample_ids)

    # prompts = [prompt for prompt in prompts if prompt["idx"] not in skipped_prompt_ids]
    new_prompts = []
    for prompt in prompts:
        if (prompt["idx"], prompt["sample_idx"]) not in skipped_prompt_sample_ids:
            new_prompts.append(prompt)
            skipped_prompt_sample_ids.add((prompt["idx"], prompt["sample_idx"]))
    prompts = new_prompts

    while len(prompts) % batch_size != 0:
        prompts.insert(0, prompts[0])

    dp_rank = get_data_parallel_rank()
    tp_rank = get_model_parallel_rank()

    dp_size = get_data_parallel_world_size()

    if tp_rank == 0:
        output_writer = open(output_file, "a")

    batch_idx = 0

    gc.collect()
    torch.cuda.empty_cache()

    max_seq_len = prompts[-1]["full_seq_len"] + 2
    print("Max sequence length:", max_seq_len)
    print("Max vocab size:", model.backbone_model.config.vocab_size)

    if compile:
        remove_all_backward_hooks(model)

    for batched_prompt_idx in range(0, len(prompts), batch_size):
        batch_idx += 1
        if batch_idx % dp_size != dp_rank:
            continue

        batched_prompts = prompts[batched_prompt_idx : batched_prompt_idx + batch_size]


        #for prompt in batched_prompts:
        #    prompt["full_seq"] = prompt["full_seq"].replace("\n\n", "<unk_token>")
        if "Meta-Llama-3" in str(checkpoint_path):
            encoded = batch_encode_tokens_llama3(
                tokenizer,
                [_["full_seq"] for _ in batched_prompts],
                bos=True,
                eos=True,
                device=device,
                padding_side="right",
            )

        elif "Llama-3.1" in str(checkpoint_path):
            encoded = batch_encode_tokens_llama3(
                tokenizer,
                [_["full_seq"] for _ in batched_prompts],
                bos=True,
                eos=True,
                device=device,
                padding_side="right",
            )

        else:
            encoded = batch_encode_tokens(
                tokenizer,
                [_["full_seq"] for _ in batched_prompts],
                bos=True,
                eos=True,
                device=device,
                padding_side="right",
            )
        #print("/n/n", tokenizer.encode("\n\n", bos=False, eos=False))
        #print("/n/n", tokenizer.encode("\n\n", bos=True, eos=True))
        #print("encoded", encoded[0])
        prompt_length = encoded.size(1)

        model_vocab_size = model.backbone_model.config.vocab_size
        #print("model_vocab_size", model_vocab_size)
        encoded[encoded >= model_vocab_size] = model_vocab_size - 1

        # torch.cuda.synchronize()
        t0 = time.perf_counter()


        try:

            y = model_score(
                model,
                encoded,
                max_seq_len=max_seq_len,
            )

        except torch.cuda.OutOfMemoryError:
            sys.exit(0)

        # import pdb

        # pdb.set_trace()

        assert y.size(0) == len(batched_prompts)
        assert y.size(1) == prompt_length

        inputs = encoded.tolist()
        outputs = y.tolist()

        num_double_newlines = [_["full_seq"].count("\n\n") for _ in batched_prompts]


        if "Meta-Llama-3" in str(checkpoint_path):
            num_tokenized_double_newlines = [
                (", ".join(str(_) for _ in encoded_full_seq) + ",").count(" 128011, 128011,")
                for encoded_full_seq in inputs
            ]

        elif "Llama-3.1" in str(checkpoint_path):
            num_tokenized_double_newlines = [
                (", ".join(str(_) for _ in encoded_full_seq) + ",").count(" 128011, 128011,")
                for encoded_full_seq in inputs
            ]


        elif "deepseek-math" in str(checkpoint_path):
            num_tokenized_double_newlines = [
                (", ".join(str(_) for _ in encoded_full_seq) + ",").count(" 185, 185,")
                for encoded_full_seq in inputs
            ]

        else:
            num_tokenized_double_newlines = [
                (", ".join(str(_) for _ in encoded_full_seq) + ",").count(" 13, 13,")
                for encoded_full_seq in inputs
            ]

        #print("inputs", inputs[0])

        #print("prompts", batched_prompts[0])
        assert len(num_double_newlines) == len(num_tokenized_double_newlines), (
            num_double_newlines,
            num_tokenized_double_newlines,
        )

        for i in range(len(num_double_newlines)):
            assert num_double_newlines[i] == num_tokenized_double_newlines[i], (
                num_double_newlines[i],
                num_tokenized_double_newlines[i],
                batched_prompts[i],
            )

        all_outputs = [_["output"] for _ in batched_prompts]
        all_prompts = [_["prompt"] for _ in batched_prompts]

        if "Meta-Llama-3" in str(checkpoint_path):
            packed_post_process = [
            post_process_newline_scores_llama3(
                inputs[i],
                outputs[i],
                all_prompts[i],
                all_outputs[i],
                process_reward_with_answer,
            )
            for i in range(len(outputs))
            ]

        elif "Llama-3.1" in str(checkpoint_path):
            packed_post_process = [
            post_process_newline_scores_llama3(
                inputs[i],
                outputs[i],
                all_prompts[i],
                all_outputs[i],
                process_reward_with_answer,
            )
            for i in range(len(outputs))
            ]
            
        elif "deepseek-math" in str(checkpoint_path):
            packed_post_process = [
            post_process_newline_scores_deepseek(
                inputs[i],
                outputs[i],
                all_prompts[i],
                all_outputs[i],
                process_reward_with_answer,
            )
            for i in range(len(outputs))
            ]           
        else:


            packed_post_process = [
                post_process_newline_scores(
                inputs[i],
                outputs[i],
                all_prompts[i],
                all_outputs[i],
                process_reward_with_answer,
                )
                for i in range(len(outputs))
            ]


        #print("packed_post_process", packed_post_process[0])

        new_line_scores = [_[0] for _ in packed_post_process]
        beautiful_outputs = [_[1] for _ in packed_post_process]

        print(beautiful_outputs[0])
        print()

        # torch.cuda.synchronize()
        t = time.perf_counter() - t0
        tokens_generated = prompt_length * y.size(0)
        tokens_sec = tokens_generated / t
        print(f"Prompt length: {prompt_length}")
        print(
            f"Time for inference {batched_prompt_idx + batch_size} / {len(prompts)}"
            f": {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        if tp_rank == 0:
            fcntl.flock(output_writer, fcntl.LOCK_EX)
            try:
                for prompt, score in zip(batched_prompts, new_line_scores):
                    output_writer.write(
                        json.dumps(
                            {
                                "idx": prompt["idx"],
                                "sample_idx": prompt["sample_idx"],
                                "prompt": prompt["prompt"],
                                "output": prompt["output"],
                                "reward": score,
                            }
                        )
                        + "\n"
                    )
                output_writer.flush()
            finally:
                fcntl.flock(output_writer, fcntl.LOCK_UN)


def post_process_newline_scores_llama3(
    encoded_full_seq, scores, prompt, output, process_reward_with_answer
):
    #print("encoded_full_seq", encoded_full_seq)
    encoded_full_seq = [_ for _ in encoded_full_seq if _ != 128008]
    encoded_full_seq_string = ", ".join(str(_) for _ in encoded_full_seq) + ","

    prompt_newlines = prompt.count("\n\n")
    output_newlines = output.split("# Answer")[0].count("\n\n")
    full_seq_newlines = encoded_full_seq_string.count(" 128011, 128011,")


    #print("full_seq_newlines", full_seq_newlines)

    # we get scores at the position from the end of the prompt
    # to the end of the output

    assert prompt.endswith("\n\n")
    assert prompt_newlines + output_newlines <= full_seq_newlines

    encoded_prompt_string = (
        " 128011, 128011,".join(encoded_full_seq_string.split(" 128011, 128011,", prompt_newlines)[:-1])
        + " 128011, 128011"
    )
    prompt_seq_len = len(encoded_prompt_string.split(","))
    encoded_output_string = encoded_full_seq_string.split(" 128011, 128011,", prompt_newlines)[
        -1
    ]
    splitted_encoded_full_seq_string = [
        _.strip() for _ in encoded_full_seq_string.split(",") if _.strip()
    ]

    newline_scores = []
    beautiful_output = [prompt.strip()]
    pos = prompt_seq_len

    for i, encoded_segment, segment in zip(
        range(output_newlines + 2),
        encoded_output_string.split(" 128011, 128011,"),
        output.split("\n\n"),
    ):
        pos += len([_.strip() for _ in encoded_segment.split(",") if _.strip()]) + 2
        if i < output_newlines - 1 and pos <= len(scores):
            assert splitted_encoded_full_seq_string[pos - 1] == "128011"
            assert splitted_encoded_full_seq_string[pos - 2] == "128011"
            newline_scores.append(scores[pos - 2])
            beautiful_output.append(
                segment + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
            )
        elif i == output_newlines - 1:
            assert splitted_encoded_full_seq_string[pos - 1] == "128011"
            assert splitted_encoded_full_seq_string[pos - 2] == "128011"
            if process_reward_with_answer:
                beautiful_output.append(segment)
            else:
                newline_scores.append(scores[pos - 2])
                beautiful_output.append(
                    segment
                    + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
                )
        elif i == output_newlines:
            if pos <= len(splitted_encoded_full_seq_string):
                assert splitted_encoded_full_seq_string[pos - 1] == "128011"
                assert splitted_encoded_full_seq_string[pos - 2] == "128011"
            beautiful_output.append(segment)
        elif i == output_newlines + 1:
            if process_reward_with_answer:
                #print("splitted_encoded_full_seq_string", splitted_encoded_full_seq_string)
                if splitted_encoded_full_seq_string[pos - 3] == "128001":
                    newline_scores.append(scores[pos - 4])
                    beautiful_output.append(segment + f" ({scores[pos - 4]})")
                else:
                    newline_scores.append(-5.0)
                    beautiful_output.append(segment + f" (Broken)")
            else:
                beautiful_output.append(segment)
        else:
            raise NotImplementedError

    return newline_scores, "\n\n".join(beautiful_output)
   

def post_process_newline_scores(
    encoded_full_seq, scores, prompt, output, process_reward_with_answer
):
    #print("encoded_full_seq", encoded_full_seq)
    encoded_full_seq = [_ for _ in encoded_full_seq if _ != 0]
    encoded_full_seq_string = ", ".join(str(_) for _ in encoded_full_seq) + ","

    prompt_newlines = prompt.count("\n\n")
    output_newlines = output.split("# Answer")[0].count("\n\n")
    full_seq_newlines = encoded_full_seq_string.count(" 13, 13,")

    # we get scores at the position from the end of the prompt
    # to the end of the output

    assert prompt.endswith("\n\n")
    assert prompt_newlines + output_newlines <= full_seq_newlines

    encoded_prompt_string = (
        " 13, 13,".join(encoded_full_seq_string.split(" 13, 13,", prompt_newlines)[:-1])
        + " 13, 13"
    )
    prompt_seq_len = len(encoded_prompt_string.split(","))
    encoded_output_string = encoded_full_seq_string.split(" 13, 13,", prompt_newlines)[
        -1
    ]
    splitted_encoded_full_seq_string = [
        _.strip() for _ in encoded_full_seq_string.split(",") if _.strip()
    ]

    newline_scores = []
    beautiful_output = [prompt.strip()]
    pos = prompt_seq_len

    for i, encoded_segment, segment in zip(
        range(output_newlines + 2),
        encoded_output_string.split(" 13, 13,"),
        output.split("\n\n"),
    ):
        pos += len([_.strip() for _ in encoded_segment.split(",") if _.strip()]) + 2
        if i < output_newlines - 1 and pos <= len(scores):
            assert splitted_encoded_full_seq_string[pos - 1] == "13"
            assert splitted_encoded_full_seq_string[pos - 2] == "13"
            newline_scores.append(scores[pos - 2])
            beautiful_output.append(
                segment + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
            )
        elif i == output_newlines - 1:
            assert splitted_encoded_full_seq_string[pos - 1] == "13"
            assert splitted_encoded_full_seq_string[pos - 2] == "13"
            if process_reward_with_answer:
                beautiful_output.append(segment)
            else:
                newline_scores.append(scores[pos - 2])
                beautiful_output.append(
                    segment
                    + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
                )
        elif i == output_newlines:
            if pos <= len(splitted_encoded_full_seq_string):
                assert splitted_encoded_full_seq_string[pos - 1] == "13"
                assert splitted_encoded_full_seq_string[pos - 2] == "13"
            beautiful_output.append(segment)
        elif i == output_newlines + 1:
            if process_reward_with_answer:
                if splitted_encoded_full_seq_string[pos - 3] == "2":
                    newline_scores.append(scores[pos - 4])
                    beautiful_output.append(segment + f" ({scores[pos - 4]})")
                else:
                    newline_scores.append(-5.0)
                    beautiful_output.append(segment + f" (Broken)")
            else:
                beautiful_output.append(segment)
        else:
            raise NotImplementedError

    return newline_scores, "\n\n".join(beautiful_output)


def post_process_newline_scores_deepseek(
    encoded_full_seq, scores, prompt, output, process_reward_with_answer
):
    #print("encoded_full_seq", encoded_full_seq)
    encoded_full_seq = [_ for _ in encoded_full_seq if _ != 100002]
    encoded_full_seq_string = ", ".join(str(_) for _ in encoded_full_seq) + ","

    prompt_newlines = prompt.count("\n\n")
    output_newlines = output.split("# Answer")[0].count("\n\n")
    full_seq_newlines = encoded_full_seq_string.count(" 185, 185,")

    # we get scores at the position from the end of the prompt
    # to the end of the output

    assert prompt.endswith("\n\n")
    assert prompt_newlines + output_newlines <= full_seq_newlines

    encoded_prompt_string = (
        " 185, 185,".join(encoded_full_seq_string.split(" 185, 185,", prompt_newlines)[:-1])
        + " 185, 185"
    )
    prompt_seq_len = len(encoded_prompt_string.split(","))
    encoded_output_string = encoded_full_seq_string.split(" 185, 185,", prompt_newlines)[
        -1
    ]
    splitted_encoded_full_seq_string = [
        _.strip() for _ in encoded_full_seq_string.split(",") if _.strip()
    ]

    newline_scores = []
    beautiful_output = [prompt.strip()]
    pos = prompt_seq_len

    for i, encoded_segment, segment in zip(
        range(output_newlines + 2),
        encoded_output_string.split(" 185, 185,"),
        output.split("\n\n"),
    ):
        pos += len([_.strip() for _ in encoded_segment.split(",") if _.strip()]) + 2
        if i < output_newlines - 1 and pos <= len(scores):
            assert splitted_encoded_full_seq_string[pos - 1] == "185"
            assert splitted_encoded_full_seq_string[pos - 2] == "185"
            newline_scores.append(scores[pos - 2])
            beautiful_output.append(
                segment + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
            )
        elif i == output_newlines - 1:
            assert splitted_encoded_full_seq_string[pos - 1] == "185"
            assert splitted_encoded_full_seq_string[pos - 2] == "185"
            if process_reward_with_answer:
                beautiful_output.append(segment)
            else:
                newline_scores.append(scores[pos - 2])
                beautiful_output.append(
                    segment
                    + f" ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
                )
        elif i == output_newlines:
            if pos <= len(splitted_encoded_full_seq_string):
                assert splitted_encoded_full_seq_string[pos - 1] == "185"
                assert splitted_encoded_full_seq_string[pos - 2] == "185"
            beautiful_output.append(segment)
        elif i == output_newlines + 1:
            if process_reward_with_answer:
                if splitted_encoded_full_seq_string[pos - 3] == "100001":
                    newline_scores.append(scores[pos - 4])
                    beautiful_output.append(segment + f" ({scores[pos - 4]})")
                else:
                    newline_scores.append(-5.0)
                    beautiful_output.append(segment + f" (Broken)")
            else:
                beautiful_output.append(segment)
        else:
            raise NotImplementedError

    return newline_scores, "\n\n".join(beautiful_output)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--prompt_file",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File to write rewards to, one per line.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune checkpoint path.",
    )

    parser.add_argument(
        "--resume_generation", action="store_true", help="Whether to resume generation."
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Size of tensor parallelism.",
    )

    parser.add_argument(
        "--on_the_fly_8bit_quantization",
        action="store_true",
        help="Whether to quantize after loading the model.",
    )

    parser.add_argument(
        "--process_reward_with_answer",
        action="store_true",
        help="Whether to apply process reward with answer.",
    )

    args = parser.parse_args()
    main(
        args.prompt_file,
        args.output_file,
        args.batch_size,
        args.checkpoint_path,
        args.compile,
        args.finetune_checkpoint_path,
        args.resume_generation,
        args.tensor_parallel_size,
        args.on_the_fly_8bit_quantization,
        args.process_reward_with_answer,
    )