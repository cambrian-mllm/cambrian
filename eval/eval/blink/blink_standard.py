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

def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None, model_name = None):
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category
    model_metrics = {}
    with open(output_file, 'r') as out_file:
        model_metrics = json.load(out_file)



    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)

            model = data.get('model_id', '')
            category = data.get('category', '')
            question_id = str(int(data.get('question_id', ''))+1)
            search_id = f"val_{category}_{question_id}"
            answer = model_metrics[search_id].lower()[1]
            categories.add(category) 

            if category not in category_metrics:
                category_metrics[category] = {'matches': 0, 'total': 0}

            # answer = data.get('answer', '').split('.')[0].lower().strip()
            gt_answer = data.get('gt_answer', '').lower()[1]
            
            category_metrics[category]['total'] += 1

            if answer == gt_answer:
                category_metrics[category]['matches'] += 1
                    

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
        "model": model if model_name is None else model_name,
        "time": time_string,
        **overall_metrics,
        **category_scores
    }

    add_data_to_csv(csv_file, combined_data)
    print(f"Saved {model} metrics to {csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl", help="Path to the jsonl file containing the model predictions")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the predictions in https://github.com/zeyofu/BLINK_Benchmark/blob/main/eval/saved_val_predictions/")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--model", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()
    compute_metrics(args.answers_file, args.output_file, args.csv_file, args.extra_outdir, args.model)