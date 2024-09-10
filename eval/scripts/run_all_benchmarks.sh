#!/bin/bash
set -e

echo "> run_all_benchmarks.sh $@"

ckpt="$1"
conv_mode="$2"

benchmarks=(
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    docvqa
    infovqa
    stvqa
    ocrbench
    mmstar
    realworldqa
    synthdog
)

# Create a directory for checkpoint files if it doesn't exist
checkpoint_dir="checkpoints"
mkdir -p "$checkpoint_dir"

# Generate a unique checkpoint file name based on the script arguments
checkpoint_file="$checkpoint_dir/checkpoint_$(basename $ckpt)_$conv_mode.txt"
script_dir=$(dirname $(realpath $0))

# Check if the checkpoint file exists and load the completed benchmarks
if [[ -f "$checkpoint_file" ]]; then
    completed_benchmarks=($(cat "$checkpoint_file"))
    echo "Resuming from checkpoint. Completed benchmarks: ${completed_benchmarks[@]}"
else
    completed_benchmarks=()
fi

timestamp=$(date +%Y%m%d-%H%M%S)
for benchmark in "${benchmarks[@]}"; do
    if [[ " ${completed_benchmarks[@]} " =~ " $benchmark " ]]; then
        echo "Skipping completed benchmark: $benchmark"
        continue
    fi

    echo "Running benchmark: $benchmark"
    bash $script_dir/run_benchmark.sh --benchmark $benchmark --ckpt $ckpt --conv_mode $conv_mode
    echo "Finished benchmark: $benchmark"

    # Append the completed benchmark to the checkpoint file
    echo "$benchmark" >> "$checkpoint_file"

    cur_timestamp=$(date +%Y%m%d-%H%M%S)
    echo "Elapsed minutes: $(( ($(date -d $cur_timestamp +%s) - $(date -d $timestamp +%s)) / 60 ))"
    echo ""
done

echo "Finished all benchmarks"
echo "Elapsed minutes: $(( ($(date -d $cur_timestamp +%s) - $(date -d $timestamp +%s)) / 60 ))"
