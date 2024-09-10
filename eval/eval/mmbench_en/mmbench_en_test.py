import json
import csv
import os
import pandas as pd
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

def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    model = ""
    categories = set()  # To store unique categories
    category_metrics = {}  # To store metrics for each category
    category_collect = {}  # To store circular predictions
    category_collect_count = {}

    list_data = []
    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                data = json.loads(line)
                model = data.get('model_id', '')
                category = data.get('category', '')
                source  = data.get('source_id', '')
                categories.add(category)

                if category not in category_metrics:
                    category_metrics[category] = {'matches': 0, 'total': 0}
                    category_collect[category] = {}
                    category_collect_count[category] = {}

                answer = extract_mcq_answer(data.get('prediction', ''))
                list_data.append(data)

                gt_answer = data.get('gt_answer', '')
                if (gt_answer == "") or (gt_answer is None):
                    # test set example
                    continue
                else:
                    gt_answer = gt_answer.lower()
                category_metrics[category]['total'] += 1

                if answer == gt_answer:
                    category_metrics[category]['matches'] += 1
                    if source not in category_collect[category]:
                        category_collect[category][source] = True
                        category_collect_count[category][source] = 1
                    else:
                        category_collect[category][source] = category_collect[category][source] and True
                        category_collect_count[category][source] += 1
                else:
                    category_collect[category][source] = False
                    if source in category_collect_count[category]:
                        category_collect_count[category][source] += 1
                    else:
                        category_collect_count[category][source] = 1
                    out_file.write(line)

    category_scores = {}
    total_matches = 0
    total_count = 0

    circular_count = 0
    circular_matches = 0

    for category, metrics in category_metrics.items():
        matches = metrics['matches']
        total = metrics['total']
        total_matches += matches
        total_count += total
        accuracy = (matches * 1.0 / total)
        category_scores[category] = {'accurcay': accuracy, 'total': total}

        # circular eval
        total_correct = 0
        circular_count += len(category_collect[category]) 
        for source_id, value in category_collect[category].items():
            total_correct += int(value)

        circular_matches += total_correct
        category_scores[category]['circular_accuracy'] = (total_correct * 1.0/len(category_collect[category]))
        category_scores[category]['circular_count'] = len(category_collect[category]) 
    overall_accuracy = (total_matches * 1.0 / total_count)
    circular_accuracy = (circular_matches*1.0 / circular_count)

    overall_metrics = {
        'accuracy': overall_accuracy,
        'total_count': total_count,
        'circular_accuracy': circular_accuracy,
        'circular_count': circular_count
    }

    combined_data = {
        "model": model,
        "time": time_string,
    }
    combined_data.update(overall_metrics)
    combined_data.update(category_scores)
    add_data_to_csv(csv_file, combined_data)
    print("Finished")

    submission_file = f"./{model}_mmbench_en_submission.xlsx"
    if len(list_data) != 0:
        df = pd.DataFrame(list_data)
        df.to_excel(submission_file, index=False)
        print(f"{submission_file} created, Please submit at [https://mmbench.opencompass.org.cn/mmbench-submission]\n")

    with open("./mmbench_en_submission_url.txt", "w") as f:
        f.write("https://mmbench.opencompass.org.cn/mmbench-submission")

    # add a duplicate copy of the info to the extra_outdir
    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_submission_file = os.path.join(extra_outdir, f"mmbench_en_submission_{model}.xlsx")

        if len(list_data) != 0:
            df.to_excel(extra_submission_file, index=False)

        with open(os.path.join(extra_outdir, "mmbench_en_submission_url.txt"), "w") as f:
            f.write("https://mmbench.opencompass.org.cn/mmbench-submission")

        print(f"Added a copy of the submission file to {extra_submission_file}")

        extra_csv_file = os.path.join(extra_outdir, f"mmbench_en_{model}.csv")
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
