import os
import json
import csv
from datetime import datetime

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def extract_mcq_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    
    text = text.strip().rstrip('.:,').lstrip('(').rstrip(')')
    if len(text) > 1:
        text = text[0]
    return text

def extract_single_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.')
    return text

def relaxed_accuracy(pred, gt):
    return 1 if abs(pred-gt) <= abs(gt)*0.05 else 0


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category

    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                categories.add(category) 

                if category not in category_metrics:
                    category_metrics[category] = {'matches': 0, 'total': 0}

                if data.get('type', '') == "multiple-choice":
                    answer = extract_mcq_answer(data.get('answer', ''))
                else:
                    answer = extract_single_answer(data.get('answer', ''))
                gt_answer = data.get('gt_answer', '').lower()
                category_metrics[category]['total'] += 1

                match = False
                if answer == gt_answer:
                    category_metrics[category]['matches'] += 1
                    match = True
                elif is_number(gt_answer) and is_number(answer) and relaxed_accuracy(float(gt_answer), float(answer)):
                    category_metrics[category]['matches'] += 1
                    match = True
                else:
                    pass

                if not match:
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

        category_scores[category] = {'accuracy': accuracy, 'total': total}

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
        **overall_metrics,
        **category_scores
    }

    add_data_to_csv(csv_file, combined_data)
    print(f"Saved {model} metrics to {csv_file}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"mmmu_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Added a copy of the csv file to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl", help="Path to the jsonl file containing the model predictions")
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to the output file to store the incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir)
