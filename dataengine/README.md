
Overall flow:
python generate_topics.py
python wikiflow.py
python generate_qa.py
python generate_vqa.py

Explanation:
1. Give inputs in input_fields_subfields.txt in the form of {Field}: {Subfields list}. We have generated our fields and subfields list using GPT-4 but this can be manually given

Topics generation:
2. Run generate_topics.py to generate topics. Don’t forget to replace the openai key with yours
 3. We noticed that gpt output needs postprocessing sometimes, in that case, you can use topics_postprocess1.py to clean it and store to post_x file. We tried handling multiple formats in this. Then run topics_postprocess2.py to store to original file if you think the file looks good to do. We wantedly kept it separate so that
you can view the file before overriding and maybe tweak the postprocessing as necessary so that it becomes easier for future.
5. Post this, we will have a topics folder with two jsons files, each for one field. {field}.json is dictionary of {subfield}:{topics list}

Wikidata Generation:
6. We now generate wikidata based on topics from field.json using wikiflow.py. While generating wiki data, make sure to replace the GOOGLE_API_KEY and GOOGLE_SE_ID in get_google_search_results function. Post this, we will have {subfield}.json files in wikilinks, each with dictoniary of {topic}: {list of wikilinks}. We also have data folder with underlying folders, each folder for a subfield and inside each folder, we have one file per topic, containing the data extracted from the wiki links for that topic.

Dataset Generation:
8. First we generate_qa.py. Create your own user agent and openapi key.  It's multi processing code and can handle large no. of processes. I only ran 30 examples per field for the demo, but you can run the full set. Post this, we have images folder with one folder for each field and qadata folder, with one file for each field. 
9. Post process using generate_vqa.py to ensure that the image_id and the json data are matched, this sits in this vqa folder and the images in images folder. 

