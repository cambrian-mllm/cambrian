#!/bin/bash
python generate_topics.py &&
python topics_postprocess1.py &&
python topics_postprocess2.py &&
python wikiflow.py &&
python generate_qa.py &&
python generate_vqa.py
