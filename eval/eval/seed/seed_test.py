import json
import csv
import os

from datetime import datetime


tags = {}
tags[1] = "Scene Understanding"
tags[2] = "Instance Identity"
tags[3] = "Instance Attribute"
tags[4] = "Instance Location"
tags[5] = "Instance Counting"
tags[6] = "Spatial Relation"
tags[7] = "Instance Interaction"
tags[8] = "Visual Reasoning"
tags[9] = "Text Recognition"

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_mcq_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    
    text = text.strip().rstrip('.').lstrip('(').rstrip(')')
    if len(text) > 1:
        text = text[0]
    return text


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category

    with open(jsonl_file, 'r') as file:
        for line in file:
            output_file = os.path.expanduser(output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as out_file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                categories.add(category)

                if tags[category] not in category_metrics.keys():
                    category_metrics[tags[category]] = {'matches': 0, 'total': 0}

                answer = extract_mcq_answer(data.get('answer', ''))

                gt_answer = data.get('gt_answer', '').lower()
                category_metrics[tags[category]]['total'] += 1

                if answer == gt_answer:
                    category_metrics[tags[category]]['matches'] += 1
                else:
                    out_file.write(line)

    category_scores = {}
    total_matches = 0
    total_count = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        accuracy = (matches * 1.0 / total)

        category_scores[category] = {'accurcay': accuracy, 'total': total}

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
    }
    combined_data.update(overall_metrics)
    combined_data.update(category_scores)
    add_data_to_csv(csv_file, combined_data)
    print(f"Model: {model}")
    print(f"Saved incorrect predictions to {output_file}")
    print(f"Combined data: {combined_data}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"seed_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Saved extra experiment data to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl", help="Path to the jsonl file containing the model predictions")
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to the output file to store the incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir)
