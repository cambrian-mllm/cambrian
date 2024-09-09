import json
import csv
import os
import string
from datetime import datetime

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_sentence(sentence):
    translator = str.maketrans('', '', string.punctuation)
    normalized_sentence = sentence.translate(translator).lower().strip()
    return normalized_sentence


def compare_sentences(sentence1, sentence2):
    return normalize_sentence(sentence1) == normalize_sentence(sentence2)


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def is_correct(answer, gt_answer, text_answer):
    option = answer.split('.')[0]
    if compare_sentences(answer, text_answer):
        return True
    elif option == gt_answer:
        return True
    else:
        return False

def extract_answer(text):
    text = text.lower().strip()
    answer_keywords = ["answer is", "answer is:", "answer:"]
    for answer_keyword in answer_keywords:
        if answer_keyword in text:
            text = text .split(answer_keyword)[-1]
    text = text.strip().rstrip('.')
    return text


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category
    category_metrics["is_multimodal"] = {'matches': 0, 'total': 0}

    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                qn_type = data.get('type', False)
                categories.add(category)

                if category not in category_metrics:
                    category_metrics[category] = {'matches': 0, 'total': 0}

                answer = extract_answer(data.get('answer', ''))
                gt_answer = str(data.get('gt_answer', '')).lower().strip()
                text_answer = data.get('text_answer', '').lower().strip()
                category_metrics[category]['total'] += 1
                if qn_type:
                    category_metrics['is_multimodal']['total'] += 1

                if is_correct(answer, gt_answer, text_answer):
                    category_metrics[category]['matches'] += 1
                    if qn_type:
                        category_metrics['is_multimodal']['matches'] += 1
                else:
                    out_file.write(line)

    category_total_scores = {}
    total_matches = 0
    total_count = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']

        total_matches += matches
        total_count += total

        accuracy = (matches * 1.0 / total)

        category_total_scores[category] = {'accuracy': accuracy, 'total': total}

    multimodal_accuracy = category_total_scores["is_multimodal"]["accuracy"]

    overall_accuracy = (total_matches * 1.0 / total_count)

    overall_metrics = {
        'accuracy': overall_accuracy,
        'total_count': total_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
        "multimodal_acc": multimodal_accuracy,
    }
    combined_data.update(overall_metrics)
    combined_data.update(category_total_scores)
    add_data_to_csv(csv_file, combined_data)
    print(f"Model: {model}")
    print(f"Metrics: {combined_data}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"scienceqa_{model}.csv")
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