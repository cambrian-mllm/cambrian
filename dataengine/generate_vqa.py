import argparse
import json
import os


def process_topic(topicname, image_dir, qa_dir, vqa_dir):
    # Read existing files to determine the next starting number
    os.makedirs(vqa_dir, exist_ok=True)
    image_dir = f'{image_dir}/{topicname}_images'
    json_file_path = f'{qa_dir}/{topicname}.json'
    output_json_file_path = f'{vqa_dir}/{topicname}.json'

    existing_files = os.listdir(image_dir)
    max_number = max((int(f.split('.')[0]) for f in existing_files if f.split('.')[
                     0].isdigit() and f.endswith('.png')), default=0)
    print(f'Max number is {max_number}')

    # Dictionary to hold new names
    renamed_files = {}

    # Process image files that need renaming
    img_files = [f for f in existing_files if f.startswith('img_')]
    img_files.sort(key=lambda x: (
        int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))

    # Start renaming from max_number + 1
    start_number = max_number + 1
    for i, filename in enumerate(img_files):
        new_filename = f"{start_number + i}.png"
        renamed_files[filename] = new_filename

    # Read the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Update the JSON data with new image names
    for item in data:
        img_id = f"{item['id']}.png"
        if img_id in renamed_files:
            item['id'] = renamed_files[img_id].replace('.png', '')

    # Save the updated JSON data
    with open(output_json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    # Optionally, rename the files in the directory (uncomment to use)
    for original_filename, new_filename in renamed_files.items():
        os.rename(os.path.join(image_dir, original_filename),
                  os.path.join(image_dir, new_filename))


def main(topics_dir, image_dir, qa_dir, vqa_dir):
    topic_files = [f for f in os.listdir(topics_dir) if f.endswith('.json')]
    topicnames = [os.path.splitext(f)[0] for f in topic_files]
    for topicname in topicnames:
        process_topic(topicname, image_dir, qa_dir, vqa_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process and rename images and update JSON data accordingly.')
    parser.add_argument('--topics_dir', type=str, required=True,
                        help='Path to the directory containing topic JSON files')
    parser.add_argument('--image_dir', type=str,
                        default='./data/images', help='Path to the images directory')
    parser.add_argument('--qa_dir', type=str, default='./data/qadata/',
                        help='Path to the input qa JSON file dir')
    parser.add_argument('--vqa_dir', type=str, default='./data/vqadata',
                        help='Path to the output vqa JSON file dir')
    args = parser.parse_args()

    main(args.topics_dir, args.image_dir, args.qa_dir, args.vqa_dir)
