#!/bin/bash

# Command to list TPU resources
output=$(gcloud alpha compute tpus queued-resources list \
  --project nyu-vision-lab \
  --zone us-central2-b)

echo "$output"
echo ""  # Add a blank line for readability

# Use awk to extract the numeric parts of the accelerator types and count TPUs
echo "$output" | awk '
BEGIN {
  sum = 0
  count = 0
}
$4 ~ /^v4-/ {
  split($4, a, "-")
  sum += a[2]
  count++
}
END {
  print "Total num TPU nodes:", sum, "/ 1152."
  print "\t=>", 1152 - sum, "remaining."
  print "Count of TPUs:", count
}
'