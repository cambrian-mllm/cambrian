# Overall Workflow
- `python generate_topics.py`
- `python wikiflow.py`
- `python generate_qa.py`
- `python generate_vqa.py`

## Explanation

### Input Specification
Provide inputs in `input_fields_subfields.txt` in the format `{Field}: {Subfields list}`. These can be generated using GPT-4 or manually specified.

### Topics Generation
2. Execute `generate_topics.py` to generate topics. Remember to replace the OpenAI key with your own.
3. GPT output sometimes requires postprocessing. In such cases, use `topics_postprocess1.py` to clean the data and store it in `post_x` files. Multiple formats can be handled.
4. Optionally, run `topics_postprocess2.py` to save the cleaned data back to the original file if the modifications are satisfactory.
5. After processing, the topics will be saved in a folder with two JSON files, each for one field. The format is `{field}.json` containing a dictionary of `{subfield}:{topics list}`.

### Wikidata Generation
6. Use `wikiflow.py` to generate wikidata based on topics from `field.json`. Be sure to update the `GOOGLE_API_KEY` and `GOOGLE_SE_ID` in the `get_google_search_results` function.
7. The output will be `{subfield}.json` files containing dictionaries of `{topic}: {list of wikilinks}`. Each subfield will have its folder with individual files for each topic, containing data extracted from the wiki links.

### Dataset Generation
8. Start by running `generate_qa.py` with your own user agent and OpenAI key. This script is designed for multiprocessing and can handle a large number of processes. Initially, 30 examples per field were run for demonstration, but it can be scaled up.
9. Post-processing is done with `generate_vqa.py` to ensure that `image_id` and JSON data are correctly matched. This data is stored in the `vqa` folder, with associated images in the `images` folder.