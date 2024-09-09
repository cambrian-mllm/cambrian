#!/bin/bash
set -e

# add help statement
if [[ $# -eq 0 || $1 == "-h" || $1 == "--help" ]]; then
    echo 'Usage: consolidate_gcp.bash <gcp_link>'
    exit 1
fi

# 1 arg = gcp link
gcp_link=$1

# eg: gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-DFN-CLIP-737k-bs512-res224

# if doesnt start with gcp, then its a local path
if [[ $gcp_link != gs* ]]; then
    echo "Not a gs link. Recieved: $gcp_link"
    exit 1
fi

# if ends with /, remove it
if [[ $gcp_link == */ ]]; then
    gcp_link=${gcp_link::-1}
fi

# extract the filename from the last /
filename=$(echo $gcp_link | rev | cut -d'/' -f1 | rev)

# ensure prepended with "cambrian-" filename to work with evals
if [[ $filename != cambrian* ]]; then
    echo "Prepending cambrian- to $filename"
    filename="cambrian-$filename"
fi


# 1. download the file
# get the checkpoint dir from env. default to ./checkpoints

ckpt_path=${CHECKPOINT_DIR:-./checkpoints}/$filename
echo "Downloading from $gcp_link to $ckpt_path"
gcloud alpha storage rsync $gcp_link $ckpt_path

# done downloading
echo "Downloaded to $ckpt_path."


# 2. now run the consolidate script
echo "Consolidating $ckpt_path"
# python scripts/scripts/consolidate.py --ckpt_path $ckpt_path --skip_existing
cmd="python scripts/consolidate.py --ckpt_path $ckpt_path --skip_existing"
echo "Running: $cmd"
$cmd
echo "Consolidated $ckpt_path"


# get the llm_model_name from env. default to lmsys/vicuna-7b-v1.5
llm_model_name=${LLM_MODEL_NAME:-"lmsys/vicuna-7b-v1.5"}

# 3. convert_hf_model
echo "Converting $ckpt_path to HF model"
# python scripts/convert_hf_model.py --ckpt_path $ckpt_path --llm_model_name $llm_model_name
cmd="python scripts/convert_hf_model.py --ckpt_path $ckpt_path --llm_model_name $llm_model_name"
echo "Running: $cmd"
$cmd
echo "Converted $ckpt_path to HF model"

