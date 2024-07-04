import argparse
import json
import os

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build


S = requests.Session()
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_SE_ID = os.getenv('GOOGLE_SE_ID')


def preprocess_google(google_output):
    preprocessed = []
    for item in google_output:
        preprocessed.append({
            "title": item.get('title', ''),
            "url": item.get('link', ''),
            "description": item.get('snippet', '')
        })
    return preprocessed


def get_google_search_results(search_query):
    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        if 'items' in res:
            return res['items']
        else:
            return []  # Return an empty list if no items are found

    results = google_search(search_query, GOOGLE_API_KEY, GOOGLE_SE_ID, num=10)
    return results


def format_results_for_query(results):
    formatted_results = []
    for result in results:
        summary = f"Title: {result['title']}, URL: {result['url']}, Description: {result['description']}"
        formatted_results.append(summary)
    return formatted_results


def just_links(results):
    links = []
    for result in results:
        links.append(result['url'])
    return links


def just_titles(results):
    titles = []
    for result in results:
        titles.append(result['title'])
    titles = [s.replace(" - Wikipedia", "") for s in titles]
    return titles


def scrape_wikipedia(pagename, field, subfield, topic, url):
    PARAMS = {
        "action": "parse",
        "page": pagename,
        "format": "json",
    }
    R = S.get(url=WIKI_API_URL, params=PARAMS)
    DATA = R.json()
    if 'error' in DATA:
        return

    soup = BeautifulSoup(DATA["parse"]["text"]["*"], 'html.parser')
    data = []

    # Wikipedia sections under different h tags
    headings = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
    skip_section = False  # Flag to skip history sections
    for i in range(len(headings)-1):
        heading = headings[i]
        next_heading = headings[i + 1]
        section_title = heading.find(
            class_='mw-headline').text if heading.find(class_='mw-headline') else 'No title'
        if 'history' in section_title.lower():
            skip_section = True
        elif heading.name == 'h2':  # Reset the flag when moving to a new main section
            skip_section = False
        if skip_section:
            continue
        content = ''
        images = []

        # Start from the current heading and end before the next heading
        element = heading.find_next_sibling()
        while element and element != next_heading:
            if element.name == 'p':
                content += ' ' + element.get_text(" ", strip=True)
            elif element.name == 'figure' or (element.name == 'div' and 'thumb' in element.get('class', [])):
                img = element.find('img')
                if img and 'src' in img.attrs:
                    # Ensure the image URL is complete
                    image_url = "https:" + img['src']
                    # Try to find a figcaption first
                    caption_el = element.find('figcaption')
                    if caption_el is None:
                        # If not found, try to find the caption in a div with class 'thumbcaption'
                        caption_el = element.find(
                            'div', class_=lambda x: x and 'thumbcaption' in x)
                    image_caption = caption_el.get_text(
                        " ", strip=True) if caption_el else "No caption"
                    images.append({'url': image_url, 'caption': image_caption})

            element = element.find_next_sibling()

        if content and images:
            data.append({
                'section': section_title,
                'text': content.strip(),
                'images': images,
                'link': url,
                # 'title': title,
                'field': field,
                'subfield': subfield,
                'topic': topic
            })

    return data


def read_topics_from_file(file_path):
    with open(file_path, 'r') as file:
        topics_data = json.load(file)
    return topics_data


def append_data_to_file(topic, subtopic, new_data, file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.extend(new_data)
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def write_datalinks_to_file(subfield, subtopic, data, path):
    os.makedirs(path, exist_ok=True)
    file_path = f'{path}{subfield}.json'
    subfield_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            subfield_data = json.load(file)
    subfield_data[subtopic] = data
    with open(file_path, 'w') as file:
        json.dump(subfield_data, file, ensure_ascii=False, indent=4)


def read_links_from_json_file(file_path, topic):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Returns an empty list if the topic is not found
    links = data.get(topic, [])
    return links


def main(topics_dir, links_dir, data_dir):
    # Modify the directory as needed
    # topics_directory = 'topics'
    # links_path = 'wikidata/wikilinks/'
    # data_path = 'wikidata/data/'
    json_files = [os.path.join(topics_dir, f) for f in os.listdir(
        topics_dir) if f.endswith('.json')]
    os.makedirs(links_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    skip = False
    print(json_files)
    for json_file in json_files:
        topics = read_topics_from_file(json_file)
        file_base_name = os.path.splitext(os.path.basename(json_file))[
            0]  # Extract base name without extension
        if not skip:
            for topic, subtopics in topics.items():
                for subtopic in subtopics:
                    # Define the path to check if the file exists
                    directory_path = f'{data_dir}{topic}'
                    os.makedirs(directory_path, exist_ok=True)
                    file_path = f'{directory_path}/{subtopic}.json'
                    # Check if the file exists; if not, proceed with the following operations
                    if os.path.exists(file_path):
                        print(f'skipping this file_path: {file_path}')
                        continue
                    print(f'running this file_path: {file_path}')
                    google_output = get_google_search_results(
                        f"Wikipedia {subtopic}")
                    if len(google_output) == 0:
                        print(
                            f'skiping {file_path} because of no results from google')
                        continue
                    preprocessed_google = preprocess_google(google_output)
                    # formatted_google_results = format_results_for_query(
                    #     preprocessed_google)
                    links = just_links(preprocessed_google)
                    titles = just_titles(preprocessed_google)
                    links = [link for link in links]
                    titles = [title for title in titles]
                    totaldataset = []
                    for i, title in enumerate(titles):
                        dataset = scrape_wikipedia(
                            title, file_base_name, topic, subtopic, links[i])
                        if dataset is not None:
                            totaldataset.extend(dataset)
                    write_datalinks_to_file(topic, subtopic, links, links_dir)
                    append_data_to_file(
                        topic, subtopic, totaldataset, file_path)
        else:
            for topic, subtopics in topics.items():
                for subtopic in subtopics:
                    links = read_links_from_json_file(
                        f'{links_dir}{file_base_name}.json', subtopic)
                    titles = [title for title in titles]
                    # print(links)
                    totaldataset = []
                    for title in titles:
                        dataset = scrape_wikipedia(
                            title, file_base_name, topic, subtopic)
                        totaldataset.extend(dataset)
                    write_datalinks_to_file(topic, subtopic, links, links_dir)
                    append_data_to_file(
                        topic, subtopic, totaldataset, file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Wikipedia data for various topics.')
    parser.add_argument('--topics_directory', type=str, default='./data/topics',
                        help='Directory containing topics JSON files')
    parser.add_argument('--links_dir', type=str,
                        default='./data/wikidata/wikilinks/', help='Directory to store links data')
    parser.add_argument('--data_dir', type=str, default='./data/wikidata/data/',
                        help='Directory to store processed data')
    args = parser.parse_args()

    main(args.topics_directory, args.links_dir, args.data_dir)
