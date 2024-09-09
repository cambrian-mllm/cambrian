import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import csv
import tarfile
from PIL import Image

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config, images):
    qs = line["question"] + " Options:"
    options = line["candidates"]
    for i in range(len(options)):
        option = chr(ord('A')+i)
        text = options[i]
        qs += f"\n{option}. {text}"

    qs += f"\n{args.question_extension}"

    image_path = images[line["img_path"]]
    input_image = Image.open(image_path).convert('RGB')

    if input_image is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if input_image is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = input_image
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)
    

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # get images paths
    images = {}
    file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="images_llvisionqa.tar", repo_type="dataset")
    extract_dir = os.path.dirname(file_path)
    if not os.path.exists(extract_dir+"/images"):
        # extract
        with tarfile.open(file_path, "r:") as tar:
            # Extract all contents to the directory where the tar file is located
            tar.extractall(path=extract_dir)
    
    files_in_dir = os.listdir(extract_dir+"/images/")
    for file in files_in_dir:
        images[file] = extract_dir+"/images/"+file

    # questions
    questions = []
    dev_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_dev.json", repo_type="dataset")
    # test_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_test.json", repo_type="dataset")
    
    with open(dev_file_path, "r") as json_file:
        questions.extend(json.load(json_file))
    # with open(test_file_path, "r") as json_file:
    #     questions.extend(json.load(json_file))
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        
        input_ids, image_tensor, image_sizes, prompt = process(line, args, tokenizer, image_processor, model.config, images)
        answer = line["correct_ans"] 
        options = line["candidates"]
        reverse_options = {}
        for ind, option in enumerate(options):
            reverse_options[option] = chr(ord('A')+ind)
        #gt_answer
        gt_answer = reverse_options[answer]
        qn_type = line["type"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "type": qn_type,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)

