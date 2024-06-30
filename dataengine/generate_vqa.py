import argparse
import json
import os


def main(topicname, path_to_images, qa_path, vqa_path):
    # Read existing files to determine the next starting number
    os.makedirs(vqa_path, exist_ok=True)
    path_to_images = f'{path_to_images}/{topicname}_images'
    json_file_path = f'{qa_path}/{topicname}.json'
    output_json_file_path = f'{vqa_path}/{topicname}.json'

    existing_files = os.listdir(path_to_images)
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
        os.rename(os.path.join(path_to_images, original_filename),
                  os.path.join(path_to_images, new_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process and rename images and update JSON data accordingly.')
    parser.add_argument('--topicname', type=str,
                        default='Geology_and_Earth_Sciences', help='Name of the topic to process')
    parser.add_argument('--path_to_images', type=str,
                        default='images', help='Path to the images directory')
    parser.add_argument('--qa_path', type=str, default='qadata',
                        help='Path to the input qa JSON file')
    parser.add_argument('--vqa_path', type=str, default='vqadata',
                        help='Path to the output vqa JSON file')
    args = parser.parse_args()

    main(args.topicname, args.path_to_images, args.qa_path, args.vqa_path)
