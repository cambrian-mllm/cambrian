# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import numpy as np
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import transformers
import tokenizers

import cambrian

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from cambrian.train.cambrian_trainer import CambrianTrainer

from cambrian import conversation as conversation_lib

from cambrian.utils import IS_XLA_AVAILABLE
from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3
from cambrian.train.wandb_nan_alert_callback import NanInfAlertWandbCallback
from cambrian.model import CambrianLlamaForCausalLM, CambrianMistralForCausalLM
from cambrian.model.language_model.cambrian_phi3 import CambrianPhi3ForCausalLM
from PIL import Image

from ezcolorlog import root_logger as logger

from packaging import version


logger.setLevel(logging.WARNING)





local_rank = None

XLA_DISABLE_FUNCTIONALIZATION = bool(os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))

PRINT_LOGS = True


def print_rank0(*args):
    if local_rank in (0, -1) and PRINT_LOGS:
        print(*args)


def log_rank0(log):
    if local_rank in (0, -1) and PRINT_LOGS:
        logger.info(log, stacklevel=2)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_aux_list: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default='[576]')
    connector_depth: Optional[int] = field(default=1)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=16)
    stride_of_vision_sampler_layers: Optional[int] = field(default=1)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_position: int = 35  # depends on v1 conv


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_sampler_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None

    # sanity check arg
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The total batch size for training. If passed, will be used to check that the "
                          "`per_device_train_batch_size` is set correctly."}
    )

    # GCSFS
    gcp_project: Optional[str] = field(default=None)
    """Can also set GCP_PROJECT environment variable."""
    gcs_output_dir: Optional[str] = field(default=None)
    """gs://<bucket>/<prefix>"""

    train_continue: bool = False
    resume_from_checkpoint: Optional[str] = ""

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_tower_aux', 'vision_resampler', 'vision_sampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    output_dir = os.path.join('checkpoints', output_dir.split(os.sep)[-1])
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "mm_projector")
        
        os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        ckpt = {
            'model': weight_to_save,
            'shard_metadata': trainer.model.get_shard_metadata()
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        print(f'checkpoint saved to {ckpt_path}\n', end='')
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    trainer._save(output_dir)
   
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()
        if prompt.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            prompt = prompt[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        conversations.append(prompt)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token_llama3(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|eot_id|>"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split("<|eot_id|>")
        
        cur_len = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            rou += sep
            
            # System Prompt
            if i == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # User Prompt
            elif i % 2 == 1:
                if i==1 and has_image:
                    round_len = len(tokenizer_image_token_llama3(rou, tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # Model Reponse
            elif i % 2 == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + 3] = IGNORE_INDEX
                cur_len += round_len

            
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}


    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_phi3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.PHI3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            if i != 0: # remove the first \n token
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_args = data_args
        self.length = self._get_length()

    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, 'r') as file:
            for i, _ in enumerate(file):
                pass
        return i + 1

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.length


    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, 'length_list') and hasattr(self, 'modality_length_list'):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list

        self.length_list = []
        self.modality_length_list = []
        with open(self.data_path, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                img_tokens = self.data_args.image_token_len if self._has_image(sample) else 0
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                self.length_list.append(cur_len + img_tokens)
                modality_len = cur_len if 'image' in sample else -cur_len
                self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:
        return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #sources = self.list_data_dict[i]

        with open(self.data_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx == i:
                    sources = json.loads(line.strip())
                    break
        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)
        if has_image:
            image_file = dat['image']
            image_folder = self.data_args.image_folder
            processor_aux_list = self.data_args.image_processor_aux_list
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except:
                return self.__getitem__(0)
            image_size = image.size
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result
            if self.data_args.image_aspect_ratio != 'pad':
                raise NotImplementedError("Only pad is supported for now.")
            
            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                target_resolution = processor_aux.crop_size['height']
                image_aux =  expand2square(image_aux, tuple(int(x*255) for x in processor_aux.image_mean)).resize((target_resolution, target_resolution))
                image_aux = processor_aux.preprocess(image_aux, return_tensors='pt')['pixel_values'][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        if (data_dict['labels']!=IGNORE_INDEX).sum()==0:
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict['image_aux_list'] = image_aux_list
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            data_dict['image_aux_list'] = [torch.zeros(3, processor_aux.crop_size['height'], processor_aux.crop_size['width']) for processor_aux in processor_aux_list]
            image_size = (crop_size, crop_size)
        data_dict['image_size'] = image_size
        return data_dict

def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0

def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = torch.ones(num_tokens_per_side, num_tokens_per_side+1, dtype=torch.bool)
    else:
        attention_mask = torch.ones(num_tokens_per_side, num_tokens_per_side, dtype=torch.bool)
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset((num_tokens_per_side, num_tokens_per_side), image_size)
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset-1:-1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0)-1
    return attention_mask, position_ids

    

def prepare_multimodal_data(input_ids, labels, attention_mask, image_sizes, image_token_len=576, image_aux_token_len_list=[192*192], max_length=2048):
    input_ids_im_replaced = []
    labels_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
    base_image_token_len_per_side = int(image_token_len**0.5)
    image_aux_token_len_per_side_list = [int(image_aux_token_len_per_side**0.5) for image_aux_token_len_per_side in image_aux_token_len_list]
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        assert num_images == 1, num_images
        image_size = image_sizes[batch_idx]
        
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []
        
        cur_labels = labels[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]+1])
            cur_labels_im_replaced.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_attention_mask_im_replaced.append(cur_attention_mask[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_position_ids_im_replaced.append(torch.arange(index, index+image_token_indices[i+1]-(image_token_indices[i]+1), dtype=torch.long, device=cur_input_ids.device))
            index += image_token_indices[i+1]-(image_token_indices[i]+1)
            
            if i < len(image_token_indices) - 2:
                num_tokens_per_side = int(image_token_len**0.5)
                image_token_len_with_newline = image_token_len + num_tokens_per_side
                cur_input_ids_im_replaced.append(torch.full((image_token_len_with_newline-1,), 0, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                cur_labels_im_replaced.append(torch.full((image_token_len_with_newline,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_im_attention_mask, cur_im_position_ids = prepare_image_info(image_size, image_token_len, newline=True)

                for aux_i, image_aux_token_len_per_side in enumerate(image_aux_token_len_per_side_list):
                    assert image_aux_token_len_per_side >= base_image_token_len_per_side
                    num_base_crops_per_aux_side = image_aux_token_len_per_side//base_image_token_len_per_side

                    cur_im_aux_attention_mask, _ = prepare_image_info(image_size, image_aux_token_len_per_side**2)
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.view(base_image_token_len_per_side, num_base_crops_per_aux_side, base_image_token_len_per_side, num_base_crops_per_aux_side)
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.permute(0, 2, 1, 3).contiguous().flatten(0,1).flatten(1,2)
                    cur_im_aux_attention_mask[cur_im_aux_attention_mask.sum(dim=1) == 0] = True
                    im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
                cur_im_position_ids += index
                
                if cur_attention_mask[image_token_indices[i+1]]:
                    cur_attention_mask_im_replaced.append(cur_im_attention_mask)
                    cur_position_ids_im_replaced.append(cur_im_position_ids.to(torch.long))
                    index = cur_im_position_ids.max()+1
                else:
                    num_tokens_per_side = int(image_token_len**0.5)
                    image_token_len_with_newline = image_token_len + num_tokens_per_side
                    cur_attention_mask_im_replaced.append(torch.full((image_token_len_with_newline,), 0, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype))
                    cur_position_ids_im_replaced.append(torch.full((image_token_len_with_newline,), 0, device=cur_input_ids.device, dtype=torch.long))
        
        input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
        labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
        attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
        position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))
    
    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = torch.stack(new_input_ids)
    new_labels = torch.stack(new_labels)
    new_attention_mask = torch.stack(new_attention_mask)
    new_position_ids = torch.stack(new_position_ids)
    im_aux_attention_masks_list = [torch.stack(im_aux_attention_masks) for im_aux_attention_masks in im_aux_attention_masks_list]
    return new_input_ids, new_labels, new_attention_mask, new_position_ids, im_aux_attention_masks_list


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list
    image_position: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side 

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)

        if padding_side == "left":
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, ( max_length - t.shape[0], 0), 'constant', IGNORE_INDEX) for t in labels]
        else:
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', IGNORE_INDEX) for t in labels]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position+1:] = input_ids[i, image_position:-1]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position+1:] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position+1:] = attention_mask[i, image_position:-1]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
        image_sizes = [instance['image_size'] for instance in instances]
        new_input_ids, new_labels, new_attention_mask, new_position_ids, im_aux_attention_masks_list = prepare_multimodal_data(input_ids, labels, attention_mask, image_sizes, image_token_len, image_aux_token_len_list, max_length)
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list
        )

        if 'image_aux_list' in instances[0]:
            image_aux_list = [instance['image_aux_list'] for instance in instances]
            image_aux_list = [list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)]
            if all(x is not None and x.shape == image_aux_list[0][0].shape for x in image_aux_list[0]):
                batch['images'] = [torch.stack(image_aux) for image_aux in image_aux_list]
            else:
                batch['images'] = image_aux_list

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator_kwargs = {
            'tokenizer': tokenizer,
        }

    if hasattr(data_args, 'image_token_len'):
        data_collator_kwargs['image_token_len'] = data_args.image_token_len

    if hasattr(data_args, 'vision_tower_aux_token_len_list'):
        data_collator_kwargs['image_aux_token_len_list'] = data_args.vision_tower_aux_token_len_list
    else:
        data_collator_kwargs['image_aux_token_len_list'] = [data_args.image_token_len]

    if hasattr(data_args, 'image_position'):
        data_collator_kwargs['image_position'] = data_args.image_position

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# TPU Note:The TorchXLA FSDP only takes in FP32 weight. This will create an issue when you load a very large model (>30b params) on TPU in FP32. 
# TPU-V4, for example, has 100GB of memory, and a 30b model will take up at least 120GB of memory. So the solution here is to load the model in bf16.
# Then, we rewrote the FSDP sharding code to convert the bf16 weights to FP32 weights only when shard the weight. Hence, we can use minimal memory to load and shard the model on TPU.

import torch_xla
import os
XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


@torch.no_grad()
def _shard_parameters_(self, params_to_shard) -> None:
    """
    At initialization we wrap a module with full parameters and shard the
    parameters in-place. Sharding is implemented by viewing each parameter
    as a 1D Tensor and retaining only a single slice, where the slice size
    is determined by the number of data parallel workers.

    Wrapping modules with many small parameters (or with a very large data
    parallel world size) will result in many small parameter shards and slow
    performance. In this case it's better to set *``flatten_parameters``* to
    ``True``, so that all of the small parameters in the module are combined
    into a single contiguous Tensor and sharded once.

    After this initial sharding is complete, the user can initialize a
    ``torch.optim.Optimizer`` in the usual way, i.e.::

    .. code-block:: python

        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    The optimizer will see only a single slice of parameters and will thus
    allocate less memory for optimizer state, avoiding redundancy across
    data parallel workers.

    Note: this method is implemented in a different manner from
    ``fairscale.nn.FullyShardedDataParallel``. Here we delete the original
    module parameters and create new sharded parameter tensors (instead of
    making sharded tensors an attribute of the original parameters). This
    make it easier to handle things (e.g. freeing parameters) on XLA.
    """

    #print_rank0("I actually use this to shard models!")
    if len(params_to_shard) > 0:
      # When freeing the full parameters, we point their internal XLATensor to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, dtype=self.compute_dtype, device=self.xla_device)

    # get the module names of each full parameter to shard
    params_to_shard_set = set(params_to_shard)
    assert len(params_to_shard_set) == len(params_to_shard), \
        "params_to_shard should not have dups"
    full_param_infos = []
    shared_full_param_memo = {}
    shared_full_param_infos = []
    full_params = []
    for module_name, m in self.named_modules():
      for n, p in m.named_parameters(recurse=False):
        if p.dtype != torch.float32:
          #raise TypeError("only fp32 parameters are supported")
          p.data = p.data.to(torch.float32)
        if p in params_to_shard_set:
          if p in shared_full_param_memo:
            mname, shared_m, shared_n = shared_full_param_memo[p]
            shared_full_param_infos.append(
                (module_name, mname, m, n, shared_m, shared_n))
          else:
            shared_full_param_memo[p] = (module_name, m, n)
            full_param_infos.append((module_name, m, n))
            full_params.append(p)
    assert len(full_params) == len(params_to_shard_set), \
        f"there are parameters in params_to_shard not belonging to this module."
    del shared_full_param_memo
    self.full_params = full_params
    self.full_param_infos = full_param_infos
    self.shared_full_param_infos = shared_full_param_infos

    # allocate and register new sharded parameters
    self.sharded_params = []
    for idx, (module_name, m, n) in enumerate(self.full_param_infos):
        p = self.full_params[idx]
        assert not hasattr(p, "_is_sharded")

        shard_data = self._get_shard(p)

        if shard_data.device != self.xla_device:
            # cast to XLA device if not already on XLA
            shard_data = shard_data.to(self.xla_device)
        p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
        p_shard._is_sharded = True
        p_shard._orig_size = p.size()
        p_shard._orig_name = f"{module_name}.{n}"
        p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
            ".", "_FSDP_SHARD_SEPARATOR_")
        self.register_parameter(p_shard._name, p_shard)
        self.sharded_params.append(p_shard)
        if p.device != self.xla_device:
            # cast to XLA device if not already on XLA
            p = p.to(self.xla_device).requires_grad_(p.requires_grad)
            # update p in full_params since id(p) changed after the casting
            self.full_params[idx] = p
        # Free the full parameter storage (here we free its internal XLATensor) but keep the tensor itself
        # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
        if XLA_DISABLE_FUNCTIONALIZATION:
            p.data = p.new_zeros(1)  # Old behavior before Functionalization.
        elif IS_XLA_AVAILABLE:
            import torch_xla
            torch_xla._XLAC._replace_xla_tensor(p, p.new_zeros(1))
        else:
            raise RuntimeError("XLA is not available")
        p._sharded_param = p_shard  # add a handle to the sharded parameter
        p._has_full_param = False
        # deregister the full parameter tensors from their modules (so that they won't
        # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
        # only the sharded parameters should appear in the FSDP model's `parameters()`)
        assert n in m._parameters
        m._parameters.pop(n)
        object.__setattr__(m, n, p)

    # also deregister the shared parameters
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
        assert n in m._parameters
        m._parameters.pop(n)
        shared_p = getattr(shared_m, shared_n)
        object.__setattr__(m, n, shared_p)

    assert len(self.sharded_params) == len(self.full_params)

if IS_XLA_AVAILABLE:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
    XlaFullyShardedDataParallel._shard_parameters_ = _shard_parameters_

def train(INDEX, attn_implementation=None):

    global local_rank
    
    log_rank0(f"Training on index {INDEX}. Local rank: {local_rank}")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # verify that the train_batch_size is set correctly
    if training_args.batch_size is not None:
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            world_size = xm.xrt_world_size()

            if training_args.per_device_train_batch_size is None:
                raise ValueError("If train_batch_size is set, per_device_train_batch_size must be set")

            if training_args.batch_size != training_args.per_device_train_batch_size * world_size:
                raise ValueError(f"train_batch_size ({training_args.train_batch_size}) must equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) * world_size ({world_size})")

            logger.warning(f"per_device_train_batch_size is correctly set to {training_args.per_device_train_batch_size} with world_size {world_size} to match train_batch_size {training_args.batch_size}")
            logger.warning(f"train_batch_size is {training_args.train_batch_size}")

    
    # TPU Note, the original LLaMA RMSNorm implementation has a bug here, the dtype conversion is not correct. It is ok in GPU but kills TPU training.
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = (self.weight * hidden_states).to(input_dtype)
        return output

    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = forward
    transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward

    def new_forward_conv(self, input):
        if self.bias is None:
            return self._conv_forward(input, self.weight, self.bias)
        return self._conv_forward(input, self.weight, self.bias.to(input.dtype))

    nn.Conv2d.forward = new_forward_conv

    def new_forward_linear(self, input):
        if self.bias is None:
            return F.linear(input, self.weight, self.bias)
        return F.linear(input, self.weight, self.bias.to(input.dtype)).to(input.dtype)

    nn.Linear.forward = new_forward_linear

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    else:
        log_rank0(f"Loading model in full precision")

    use_cohere = False
    data_args.image_token_len = model_args.image_token_len

    if model_args.vision_tower_aux_list is not None:
        # copy image_token_len and image_position to model_args
        # data_args.image_token_len = model_args.image_token_len
        model_args.image_position = data_args.image_position

        
        # Assuming model_args.model_name_or_path is a string that includes the model size
        model_name = model_args.model_name_or_path

        # Regular expression to find the number of parameters in the model's name (assuming a convention like 'ModelName-30b')
        match = re.search(r'(\d+)b', model_name)
        num_parameters_billion = float(match.group(1)) if match else 0

        # Determine if bfloat16 should be used based on the model's size
        use_bfloat16 = training_args.bf16 or num_parameters_billion > 30

        if "yi" in model_args.model_name_or_path.lower():
            use_bfloat16 = True

        elif "mistral" in model_name.lower():
            logger.warning(f"Vision tower, loading CambrianMistralForCausalLM: {model_args.model_name_or_path}")

            # replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer
            if (
                hasattr(training_args, 'fsdp_config') and
                'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
            ):
                logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
                training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["MistralDecoderLayer"]

            model = CambrianMistralForCausalLM.from_pretrained(
                model_name,
                cache_dir=training_args.cache_dir,
                do_sample=True,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward
        elif "phi-3" in model_name.lower():
            logger.warning(f"Vision tower, loading CambrianPhi3ForCausalLM: {model_args.model_name_or_path}")
            
            # replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer
            if (
                hasattr(training_args, 'fsdp_config') and
                'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
            ):
                logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with Phi3DecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
                training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Phi3DecoderLayer"]
            model = CambrianPhi3ForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=training_args.cache_dir,
                    do_sample=True,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
            )
            cambrian.model.language_model.phi3.modeling_phi3.Phi3RMSNorm.forward = forward
        else:
            logger.warning(f"Vision tower, loading CambrianLlamaForCausalLM: {model_args.model_name_or_path}")
            model = CambrianLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                do_sample=True,
                torch_dtype=(torch.bfloat16 if use_bfloat16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        logger.warning(f"No vision tower, loading pure language model: {model_args.model_name_or_path}")
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.generation_config.do_sample = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    log_rank0("Model loaded.")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        log_rank0("Using gradient checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        log_rank0("Adding LoRA adapters...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        print_rank0("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    log_rank0("Configuring tokenizer...")
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "llama_v3":
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            logger.warning(f"Conversation version {model_args.version} not found. Using default `vicuna_v1`")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # log_rank0(f"Default conversation version: {conversation_lib.default_conversation.version}")
    # print_rank0("Then it is", conversation_lib.default_conversation)

    if use_cohere:
        tokenizer.pad_token_id = 0
        print_rank0("tokenizer id is", tokenizer.pad_token_id)
    # print_rank0("tokenizer is", tokenizer)

    if model_args.vision_tower_aux_list is not None:
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()
        
        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            tune_modules = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline']
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    print_rank0('tuning {}'.format(name))
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        if training_args.bits in [4, 8]:
            log_rank0(f"Initializing vision modules in {training_args.bits}bit")
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.image_token_len = data_args.image_token_len = model_args.image_token_len
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = data_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        log_rank0(f"Initializing model in {training_args.bits}bit")
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    log_rank0("Configuring data module...")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    

    if training_args.bf16:
        model = model.to(dtype=torch.float32)

    callbacks = []

    if "wandb" in training_args.report_to:
        wandb_nan_callback = NanInfAlertWandbCallback(metrics=["loss"])
        callbacks.append(wandb_nan_callback)
        # rm wandb from training_args.report_to so it doesn't get passed to the Trainer
        training_args.report_to.remove("wandb")
        assert "wandb" not in training_args.report_to, training_args.report_to


    log_rank0("Configuring trainer...")
    trainer = CambrianTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    trainer.is_fsdp_enabled = True
    if training_args.train_continue:
        resume_from_checkpoint=training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    log_rank0(f"Training finished: {training_args.output_dir}")
    
    trainer.save_state()

    model.config.use_cache = True

    log_rank0("Saving model...")
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
