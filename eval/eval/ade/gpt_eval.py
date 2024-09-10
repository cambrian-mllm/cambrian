import argparse
import os
import json
import base64
import requests
import time

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


def gpt_answer(image_path, question, addition):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": question + addition
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()['choices'][0]['message']['content'] 
    time.sleep(2)
    return output

# image_path = "/scratch/eb3174/datasets/vision-benchmark/" + image_paths[i]
def eval_model(args):
    questions = []
    with open(args.prompt_path, 'r') as file:
        questions = json.load(file)
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")
    
    ans_file = open(answers_file, "w")

    idx = -1
    category_counts = {}
    # Relative location,Count + Relative location,Count
    limits = {"Relative location": 100,"Count + Relative location": 0, "Count": 100}
    for line in questions:
        idx = idx+1
        category = line["sub_task"]
        if category not in category_counts:
            category_counts[category] = 0
        
        # just do a eval on 100 examples in each category
        if category_counts[category]>=limits[category]:
            continue

        category_counts[category] = category_counts[category] + 1
        img_path = line["img_name"]
        if img_path.startswith("/ADE20K_2021_17_01/"):
            suffix = img_path[len("/ADE20K_2021_17_01/"):]
            img_path = args.images_path + suffix
        else:
            img_path = args.images_path + "ade20k_vprompt/relative_location/" + img_path

        question = line["prompt"]
        addition = f" {args.question_extension}"
        outputs = gpt_answer(img_path, question, addition)

        print(category, category_counts[category], outputs)
        ans_file.write(json.dumps({
            "questionId": idx,
            "prompt": line["prompt"],
            "answer": outputs,
            "gt_answer": line["answer"],
            "category": line["sub_task"],
            "options": line["choices"],
            "model_id": "gpt4o"
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--images_path", type=str, default="/scratch/eb3174/datasets/vision-benchmark/")
    parser.add_argument("--prompt_path", type=str, default="/scratch/mm12799/vision-benchmark/ade20k_validation_prompt.json")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)
