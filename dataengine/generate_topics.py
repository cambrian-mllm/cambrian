import argparse
import os

from openai import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip().split("\n")


def main(data_file_path, output_dir):
    client = OpenAI(api_key=OPENAI_API_KEY)
    lines = read_data_from_file(data_file_path)
    os.makedirs(output_dir, exist_ok=True)
    for line in lines:
        print(line)
        if ": " in line:
            topic, content = line.split(": ", 1)
            content = line
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You will be given a list of fields and subfields. For each subfield, generate a detailed list of 20 sub-topics in JSON format."},
                    {"role": "user", "content": f'{content}'}
                ]
            )
            print(completion.choices[0].message.content)
            file_path = f"{output_dir}/{topic.replace(' ', '_')}.json"
            with open(file_path, "w") as file:
                file.write(completion.choices[0].message.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate detailed lists of topics from subfields')
    parser.add_argument('--data_file_path', type=str, default='./data/input_fields_subfields.txt',
                        help='Path to the data file containing topics and subfields')
    parser.add_argument('--output_dir', type=str, default='./data/topics/',
                        help='Directory to output the resulting JSON files')
    args = parser.parse_args()

    main(args.data_file_path, args.output_dir)
