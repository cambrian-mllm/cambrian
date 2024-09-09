import json
import csv
import os
from datetime import datetime


current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}

reverse_eval_dict = {}
for key in eval_type_dict:
    for item in eval_type_dict[key]:
        reverse_eval_dict[item] = key


def add_data_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    categories = set()
    category_metrics = {}
    incorrect_predictions = []

    idx = -1
    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                idx += 1
                data = json.loads(line)
                category = data.get('category', '')
                model = data.get('model_id', '')
                categories.add(category)

                if category not in category_metrics:
                    category_metrics[category] = {'true_positives': 0.0, 'acc+': 0.0, 'total': 0.0}

                answer = data.get('answer', '').lower()
                answer = answer.rstrip('.') # remove trailing dot
                gt_answer = data.get('gt_answer', '').lower()
                category_metrics[category]['total'] += 1
                if answer == gt_answer:
                    category_metrics[category]['true_positives'] += 1
                    if idx % 2 == 1 and category_metrics[category]['acc+'] % 2 == 0:
                        pass
                    else:
                        category_metrics[category]['acc+'] += 1
                else:
                    if idx % 2 == 1 and category_metrics[category]['acc+'] % 2 == 1:
                            category_metrics[category]['acc+'] -= 1
                    out_file.write(line)

    category_scores = {}
    total_acc_plus = 0.0
    total_true_positives = 0.0
    total_count = 0.0
    total_score = 0.0

    for category, metrics in category_metrics.items():
        acc_plus = metrics['acc+']
        true_positives = metrics['true_positives']
        total = metrics['total']

        total_acc_plus += acc_plus
        total_true_positives += true_positives
        total_count += total

        acc_score = (true_positives/total)
        acc_plus_score = (acc_plus/total)
        total_score += 100.0 * (acc_score + acc_plus_score)

        category_scores[category] = {'acc_score': 100.0*acc_score, 'acc_plus_score': 100.0*acc_plus_score, 'score': 100.0*(acc_score+acc_plus_score), 'size': total}

    overall_metrics = {
        'total_score': total_score,
        'accuracy': 100.0 * (total_true_positives / total_count),
        'Perception': 0.0,
        'Cognition': 0.0,
    }

    for category, score in category_scores.items():
        overall_metrics[reverse_eval_dict[category]] += score['score']

    combined_data = {
        "model": model,
        "time": time_string,
    }
    combined_data.update(overall_metrics)
    combined_data.update(category_scores)
    add_data_to_csv(csv_file, combined_data)
    print(f"Data: {combined_data}")
    print(f"Saved incorrect predictions to {output_file}")
    print(f"Saved experiment data to {csv_file}")

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"mme_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Addded a copy of the experiment data to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl", help="Path to the jsonl file containing the model predictions")
    parser.add_argument("--output_file", type=str, default="./incorrect/incorrect.jsonl", help="Path to the output file to store the incorrect predictions")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir)
