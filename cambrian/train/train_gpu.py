import os
import sys

# Ensure the project's root directory is in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cambrian.train.train_fsdp_gpu import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation=None)
