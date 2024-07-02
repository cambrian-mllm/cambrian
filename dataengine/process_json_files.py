import json
import os
import re


def process_json_files(directory):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            print(filename)
            file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(directory, f"post_{filename}")
            # print(output_file_path)
            with open(file_path, 'r') as file:
                content = file.read()

            # Try to extract JSON using the regex for embedded JSON as before
            matches = re.findall(r'```json\n({.*?})\n```', content, re.DOTALL)

            if matches:
                # If match found, parse and save the JSON
                json_data = json.loads(matches[0])
            else:
                # If no match, parse the entire file content as JSON
                json_data = json.loads(content)

            if isinstance(json_data, dict):
                new_json_data = list(json_data.values())
                if len(new_json_data) <= 1:
                    json_data = new_json_data[0]
                else:
                    json_data = json_data

            def process_nested_json(json_data):
                processed_json = {}
                # Check if the item is a dictionary and process accordingly
                for key, value in json_data.items():
                    processed_list = [topic.replace(
                        "/", " or ") for topic in value]
                    processed_json[key] = processed_list
                return processed_json

            final = process_nested_json(json_data)

            with open(output_file_path, 'w') as json_file:
                # json.dump(processed_json, json_file, indent=4)
                json.dump(final, json_file, indent=4)

            print(f"Processed {filename} -> {output_file_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Process topics JSON files in a directory')
    parser.add_argument('--topics_dir', type=str, default='./data/topics/',
                        help='Directory of topics JSON files to process')
    args = parser.parse_args()

    process_json_files(args.topics_dir)
