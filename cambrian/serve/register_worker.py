"""
Manually register workers.

Usage:
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21002
"""

import spaces

import argparse
import requests

from ezcolorlog import root_logger as logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    parser.add_argument("--worker-name", type=str)
    parser.add_argument("--check-heart-beat", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": args.worker_name,
        "check_heart_beat": args.check_heart_beat,
        "worker_status": None,
    }
    logger.info(f"Registering worker to controller at {url} with data: {data}")
    r = requests.post(url, json=data)
    assert r.status_code == 200
    logger.info(f"Worker registered successfully.")
