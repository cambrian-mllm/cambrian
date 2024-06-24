import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# cambrian-phi3-3b
# conv_mode = "phi3"

# cambrian-8b
conv_mode = "llama_3" 

# cambrian-34b
# conv_mode = "chatml_direct"

# cambrian-13b
# conv_mode = "vicuna_v1"

def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt

import torch
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = os.path.expanduser("nyu-visionx/cambrian-8b")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

temperature = 0

while True:
    image_path = input("image path: ")
    image = Image.open(image_path).convert('RGB')
    question = input("question: ")

    input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor, model.config)
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)
