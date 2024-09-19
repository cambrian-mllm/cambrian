#!/bin/bash
set -e

echo "> run_benchmark.sh $@"

################# Parse Arguments #################

# Default values
# conv_mode="vicuna_v1"
conv_mode="llama_3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --conv_mode)
        conv_mode="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required arguments benchmark and ckpt are provided
if [[  -z "$benchmark" || -z "$ckpt" ]]; then
  echo "Error: --benchmark and --ckpt arguments are required."
  exit 1
fi

# Print the values
echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"

################# Process Arguments #################

# get the dir of this script
script_dir=$(dirname $(realpath $0))
benchmark_dir="${script_dir}/../eval/${benchmark}"
eval_file="${benchmark_dir}/${benchmark}_eval.py"
test_file="${benchmark_dir}/${benchmark}_test.py"

# verify that the benchmark exists and the eval/test files exist
if [ ! -d $benchmark_dir ]; then
    echo "Error: Benchmark directory $benchmark_dir does not exist."
fi
if [ ! -f $eval_file ]; then
    echo "Error: Eval file $eval_file does not exist."
    exit 1
fi
if [ ! -f $test_file ]; then
    echo "Error: Test file $test_file does not exist."
    exit 1
fi

# # verify that the ckpt directory exists
# if [ ! -d $ckpt ]; then
#     echo "Error: Checkpoint $ckpt does not exist."
#     exit 1
# fi

# cd to the benchmark directory
cd $benchmark_dir

# extract basename
model_basename=$(basename $ckpt)
answers_file="./answers/answers_${model_basename}.jsonl"


################# Handle Chunking #################
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo $CHUNKS


################# Run Evaluation #################

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

for IDX in $(seq 0 $((CHUNKS-1))); do
    {
        CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python $eval_file \
            --model_path "$ckpt" \
            --num_chunks "$CHUNKS" \
            --chunk_idx "$IDX" \
            --answers_file "$answers_file" \
            --conv_mode "$conv_mode"
    } &
done

wait

################# Combine Results #################

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

# check if the answers_file already exists. if it does, move it to a backup file with the current timestamp
if [ -f "$answers_file" ]; then
    mv "$answers_file" "${answers_file}.bak.$(date +%s)"
    echo "Moved existing answers file to ${answers_file}.bak.$(date +%s)"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    idx_file=./answers/answers_${model_basename}_${IDX}.jsonl
    cat "$idx_file" >> "$answers_file"
    rm "$idx_file"
done

################# Run Testing #####################

python $test_file --answers_file "$answers_file"

echo "Done evaluation and testing for $benchmark on model at $ckpt with conversation mode $conv_mode"
echo "Answers file: ${(realpath $answers_file)}"
