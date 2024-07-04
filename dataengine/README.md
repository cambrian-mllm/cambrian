# Overall Workflow

## Environment Setup
Please install the necesary packages using the [`requirements.txt`](requirements.txt) file.


## Prepare input
See [Input Specification](#input-specification) for details on how to prepare the input file, and [data/example_input_fields_subfields.txt](data/example_input_fields_subfields.txt) for an example. The example below expects the input file to be named `input_fields_subfields.txt` and placed in the `data` directory, but this can be changed via the enivronment variables.


## Example Workflow

The workflow consists of a series of Python scripts that should be executed in the following order:


### 1. set your environment / keys
```bash
OPENAI_API_KEY="your_openai_key"
GOOGLE_API_KEY="your_google_api_key"
GOOGLE_SE_ID="your_google_search_engine_id"
# https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
USER_AGENT="your_user_agent"  # Example: "Image downloader/1.0 (your email)"
```

### 2. set args for the scripts
```bash
# input
DATA_DIR="./data"
IN_FILE="${DATA_DIR}/example_input_fields_subfields.txt"

# intermediate output
TOPICS_DIR="${DATA_DIR}/topics/"
WIKI_DIR="${DATA_DIR}/wikidata/"
WIKI_LINKS_DIR="${WIKI_DIR}/wikilinks/"
WIKI_DATA_DIR="${WIKI_DIR}/data/"

# final output
IMAGE_DIR="${DATA_DIR}/images/"
QA_DIR="${DATA_DIR}/qadata/"
VQA_DIR="${DATA_DIR}/vqa/"
```

### 3. run the scripts
```bash
# generate topics and process
python generate_topics.py --data_file_path $IN_FILE --output_dir $TOPICS_DIR
python process_json_files.py --topics_dir $TOPICS_DIR
python clean_and_rename_files.py --topics_dir $TOPICS_DIR

# download from wikipedia / google
python wikiflow.py --topics_dir $TOPICS_DIR --links_dir $WIKI_LINKS_DIR --data_dir $WIKI_DATA_DIR

# generate vqa data
python generate_qa.py --topics_dir $TOPICS_DIR --data_dir $WIKI_DATA_DIR --qa_dir $QA_DIR --image_dir $IMAGE_DIR
python generate_vqa.py --topics_dir $TOPICS_DIR --qa_dir $QA_DIR --vqa_dir $VQA_DIR --image_dir $IMAGE_DIR
```

## Explanation

### Input Specification
Provide inputs in `input_fields_subfields.txt` in the format `{Field}: {Subfields list}`. These can be generated using GPT-4 or manually specified.

### Topics Generation
2. Execute `generate_topics.py` to generate topics. Remember to replace the OpenAI key with your own.
3. GPT output sometimes requires postprocessing. In such cases, use `process_json_files.py ` to clean the data and store it in `post_x` files. Multiple formats can be handled.
4. Optionally, run `clean_and_rename_files.py` to save the cleaned data back to the original file if the modifications are satisfactory.
5. After processing, the topics will be saved in a folder with two JSON files, each for one field. The format is `{field}.json` containing a dictionary of `{subfield}:{topics list}`.

### Wikidata Generation
6. Use `wikiflow.py` to generate wikidata based on topics from `field.json`. Be sure to update the `GOOGLE_API_KEY` and `GOOGLE_SE_ID` in the `get_google_search_results` function.
7. The output will be `{subfield}.json` files containing dictionaries of `{topic}: {list of wikilinks}`. Each subfield will have its folder with individual files for each topic, containing data extracted from the wiki links.

### Dataset Generation
8. Start by running `generate_qa.py` with your own user agent and OpenAI key. This script is designed for multiprocessing and can handle a large number of processes. Initially, 30 examples per field were run for demonstration, but it can be scaled up.
9. Post-processing is done with `generate_vqa.py` to ensure that `image_id` and JSON data are correctly matched. This data is stored in the `vqa` folder, with associated images in the `images` folder.

Below is the folder structure you will see after running the scripts using the example input file:

- **dataengine/**
  - **data/**
    - **images/**
        - **Geology_and_Earth_Sciences_images/**
            - `1.png`
            - `2.png`
            - `...`
        - **Renewable_Energy_and_Sustainability_images/**
            - `1.png`
            - `2.png`
            - `...`
    - **qadata/**
        - `Geology_and_Earth_Sciences.json`
        - `Renewable_Energy_and_Sustainability.json`
    - **topics/**
        - `Geology_and_Earth_Sciences.json`
        - `Renewable_Energy_and_Sustainability.json`
    - **wikidata/**
        - **data/**
            - `Biomass Energy/`
                - `Advancements in biofuel production.json`
                - `Bioliquids in energy production.json`
                - `...`
            - `Energy Storage/`
                - `...`
            - `Hydropower/`
                - `...`
            - `...`
        - **wikilinks/**
            - `Biomass Energy.json`
            - `Energy Storage.json`
            - `Hydropower.json`
            - `...`
  - `generate_qa.py`
  - `generate_topics.py`
  - `generate_vqa.py`
  - `input_fields_subfields.txt`
  - `process_json_files.py `
  - `clean_and_rename_files.py`
  - `wikiflow.py`
  - `README.md`
  - `requirements.txt`
  