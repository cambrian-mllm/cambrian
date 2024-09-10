import argparse
import glob
import io
import json
import multiprocessing
import os
import time
from xml.etree import ElementTree

import cairosvg
import requests
from openai import OpenAI
from PIL import Image
from ratelimit import limits, sleep_and_retry

RATE_LIMIT = 7500  # as per your resources
RATE_LIMIT_INTERVAL = 60  # Time interval of 1 minute
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
user_agent = os.getenv('USER_AGENT')
session = requests.Session()
session.headers.update({'User-Agent': user_agent})
num_processes = 4  # can go for 32 or 64 based on no. of cpus
processed_queue = multiprocessing.Queue()


def convert_thumb_url(url):
    parts = url.split('/')
    # Remove the 'thumb' part
    if 'thumb' in parts:
        parts.remove('thumb')
    # Detect if the last segment contains a size indication and remove it
    if 'px' in parts[-1]:
        parts.pop()  # Remove the last part, which is the size-indicated duplicate
    original_url = '/'.join(parts)
    return original_url


def download_image(session, image_url, filename, save_directory, delay=1):
    try:
        # Wait for a specified delay between requests to avoid rate limiting
        time.sleep(delay)

        response = session.get(image_url)
        response.raise_for_status()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        image_format = image_url.split('.')[-1].lower().split('?')[0]

        file_path = os.path.join(save_directory, f"{filename}.png")

        if image_format == 'svg':
            # Parse the SVG
            svg_root = ElementTree.fromstring(response.content)
            # Set the width and height if not set (necessary for rendering)
            if 'width' not in svg_root.attrib:
                svg_root.set('width', '800')
            if 'height' not in svg_root.attrib:
                svg_root.set('height', '600')
            # Create a white background rectangle
            background = ElementTree.Element(
                'rect', width='100%', height='100%', x='0', y='0', fill='white')
            # Insert the background rectangle at the beginning of the SVG file
            svg_root.insert(0, background)
            # Get the modified SVG content as a string
            modified_svg = ElementTree.tostring(
                svg_root, encoding='utf-8', method='xml')
            # Convert the modified SVG to PNG
            cairosvg.svg2png(bytestring=modified_svg, write_to=file_path)
        elif image_format in ['gif', 'jpg']:
            # Handle GIF and JPG files
            with Image.open(io.BytesIO(response.content)) as image:
                # Convert to RGBA to manage any transparency in GIFs
                image = image.convert('RGBA')
                if image_format == 'gif':
                    # Create a white canvas for GIFs
                    canvas = Image.new('RGBA', image.size, 'WHITE')
                    canvas.paste(image, mask=image)
                    image = canvas.convert('RGB')  # Convert back to RGB
                image.save(file_path, 'PNG')  # Save as PNG
        else:
            # Directly save other formats as PNG
            with open(file_path, 'wb') as f:
                f.write(response.content)
        return 1
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return 0


def get_topic_files(parent_directory, field, topics_directory):
    topic_files = []
    field_file_path = os.path.join(topics_directory, f'{field}.json')
    with open(field_file_path, 'r') as file:
        field_data = json.load(file)
        # print(field_name)
        for subfield, topics in field_data.items():
            for topic in topics:
                topic_files.append(os.path.join(
                    parent_directory, subfield, f'{topic}.json'))
    print(topic_files)
    return topic_files


def get_data(topic_files):
    all_data = []
    for file_path in topic_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            print(f'Processing {file_path}')
            # Filter sections with less than 50 words in the 'text'
            filtered_data = [item for item in data if len(
                item['text'].split()) > 50]
            all_data.extend(filtered_data)
        else:
            print(f'Skipping {file_path} as the file does not exist.')
    print(len(all_data))
    return all_data
    # return all_data[:50]


@sleep_and_retry
@limits(calls=RATE_LIMIT, period=RATE_LIMIT_INTERVAL)
def process_data_point(data_point, client, proc_index, data_point_index, image_directory):
    processed_data_points = []
    curr_data_point_index = data_point_index
    for image_index, image_info in enumerate(data_point['images']):
        image_url = image_info['url']
        image_caption = image_info['caption']
        image_id = image_url.split('/')[-1].split('.')[0]
        # Download the image
        customurl = convert_thumb_url(image_url)
        current_id = f"img_{proc_index}_{curr_data_point_index}"
        status = download_image(
            session, customurl, current_id, image_directory)
        if status == 1:
            # Generate question and answer using OpenAI
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a question and answer pair based on the image's caption, its contextual text, and associated metadata. The question should be insightful, engaging, and directly related to the image's content as described by the caption. The answer should be concise and informative, drawing upon the text and contextual details provided."
                    },
                    {
                        "role": "user",
                        "content": json.dumps(dict(
                            data_point,
                            image_url=image_url,
                            caption=image_caption
                        ))
                    }
                ]
            )
            response = completion.choices[0].message.content
            print(response)
            processed_data_point = {
                "id": current_id,
                "image_id": image_id,
                "image_url": image_url,
                "text": data_point['text'],
                "caption": image_caption,
                "section": data_point['section'],
                "subfield": data_point['subfield'],
                "field": data_point['field'],
                "topic": data_point['topic'],
                "pagelink": data_point['link'],
                "openairesponse": response
            }
            processed_data_points.append(processed_data_point)
            curr_data_point_index += 1

    return processed_data_points, curr_data_point_index


def worker(data_chunk, proc_index, result_list, image_directory):
    client = OpenAI(api_key=OPENAI_API_KEY)
    curr_data_point_index = 0
    for i, data_point in enumerate(data_chunk):
        results, curr_data_point_index = process_data_point(
            data_point, client, proc_index, curr_data_point_index, image_directory)
        for result in results:
            result_list.append(result)


def main(topics_dir, data_dir, image_directory, qa_directory):
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(qa_directory, exist_ok=True)
    fields = [
        os.path.splitext(f)[0]
        for f in os.listdir(topics_dir)
        if f.endswith('.json')
    ]
    for field in fields:
        field_image_directory = f'{image_directory}/{field}_images'
        os.makedirs(field_image_directory, exist_ok=True)
        id = len(glob.glob(os.path.join(field_image_directory, '*')))
        print(f'starting images with id {id}')
        topic_files = get_topic_files(data_dir, field, topics_dir)
        filtered_data = get_data(topic_files)
        processed_data_file_path = f'{qa_directory}/{field}.json'
        print(processed_data_file_path)
        if os.path.exists(processed_data_file_path):
            with open(processed_data_file_path, 'r') as file:
                processed_data = json.load(file)
        else:
            processed_data = []

        manager = multiprocessing.Manager()
        results = manager.list()

        # Split data into chunks for each process
        data_chunks = [filtered_data[i::num_processes]
                       for i in range(num_processes)]
        processes = []
        for index, chunk in enumerate(data_chunks):
            p = multiprocessing.Process(target=worker, args=(
                chunk, index, results, field_image_directory))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        processed_data.extend(results)

        with open(processed_data_file_path, 'w') as f:
            json.dump(list(processed_data), f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some topics.')
    parser.add_argument('--topics_dir', type=str,
                        default='./data/topics', help='Directory of topics')
    parser.add_argument('--data_dir', type=str,
                        default='./data/wikidata/data/', help='Data path')
    parser.add_argument('--image_dir', type=str,
                        default='./data/images', help='Directory to store images')
    parser.add_argument('--qa_dir', type=str,
                        default='./data/qadata', help='Directory to store q&a processed')
    args = parser.parse_args()

    main(args.topics_dir, args.data_dir,
         args.image_dir, args.qa_dir)
