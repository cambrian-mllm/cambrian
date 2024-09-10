import os
import json
import csv

from m4c_evaluator import TextVQAAccuracyEvaluator
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


def compute_metrics(jsonl_file, csv_file, extra_outdir=None):
    pred_list = []
    model = ""
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            answer = data.get('answer', '').lower()
            model = data.get('model_id','')
            gt_answer = [x.lower() for x in data.get('gt_answer', [''])]
            pred_list.append({
                "pred_answer": answer,
                "gt_answers": gt_answer,
            })
    evaluator = TextVQAAccuracyEvaluator()
    combined_data = {
        "model": model,
        "time": time_string,
        "accuracy": 100. * evaluator.eval_pred_list(pred_list),
    }
    add_data_to_csv(csv_file, combined_data)

    print(combined_data)

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"textvqa_{model}.csv")
        add_data_to_csv(extra_csv_file, combined_data)
        print(f"Saved experiment data to {extra_csv_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str, required=True, help="Path to the answers file")
    parser.add_argument("--csv_file", type=str, default="./experiments.csv", help="Path to the output csv file to store the experiment data")
    parser.add_argument("--extra_outdir", type=str, default=None, help="Path to an extra output directory in which to store a copy of the information")

    args = parser.parse_args()

    compute_metrics(args.answers_file, args.csv_file, args.extra_outdir)
