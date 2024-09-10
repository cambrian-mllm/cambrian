import os
import json
import nltk
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


def is_inflection(word1, word2):
    """
    Checks if two words are likely inflections of the same word using stemming and lemmatization.

    Args:
        word1: The first word.
        word2: The second word.

    Returns:
        True if the words are likely inflections, False otherwise.
    """
    # Lowercase both words for case-insensitive comparison
    word1 = word1.lower()
    word2 = word2.lower()

    # Use Porter stemmer for a more aggressive reduction to the base form
    stemmer = nltk.PorterStemmer()
    stem1 = stemmer.stem(word1)
    stem2 = stemmer.stem(word2)

    # Use WordNet lemmatizer for a more accurate base form considering context
    lemmatizer = nltk.WordNetLemmatizer()
    lemma1 = lemmatizer.lemmatize(word1)
    lemma2 = lemmatizer.lemmatize(word2)

    # Check if stemmed or lemmatized forms are equal
    return (stem1 == stem2) or (lemma1 == lemma2)


def compute_metrics(jsonl_file, output_file, csv_file, extra_outdir=None):
    correct, total = 0, 0
    model = ""
    with open(jsonl_file, 'r') as file:
        output_file = os.path.expanduser(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for line in file:
                total += 1.0
                data = json.loads(line)
                model = data.get('model_id', '')
                answer = data.get('answer', '').lower().rstrip('.')
                gt_answer = data.get('gt_answer', '').lower()
                if (answer == gt_answer):
                    correct += 1.0
                else:
                    out_file.write(line)

    combined_data = {
        "model": model,
        "time": time_string,
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct/total,
    }
    add_data_to_csv(csv_file, combined_data)

    if extra_outdir is not None:
        os.makedirs(extra_outdir, exist_ok=True)
        extra_csv_file = os.path.join(extra_outdir, f"gqa_{model}.csv")
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
