#!/bin/bash

# Print the list of TPU resources in any state
gcloud alpha compute tpus queued-resources list \
  --project nyu-vision-lab \
  --zone us-central2-b

echo  # Add a blank line for readability

# List all TPU resources in suspended or failed state and extract their names
SUSPENDED=$(gcloud alpha compute tpus queued-resources list \
  --project nyu-vision-lab \
  --zone us-central2-b | awk '$NF=="SUSPENDED" || $NF=="FAILED"{print $1}')

# If no resources are in suspended state, exit
if [ -z "$SUSPENDED" ]; then
  echo "No TPU VMs in suspended state."
  exit 0
fi

# Loop through each suspended resource and delete it
for NAME in $SUSPENDED; do
  echo "Deleting TPU VM: $NAME"
  gcloud alpha compute tpus queued-resources delete $NAME --project nyu-vision-lab --zone us-central2-b --quiet
done

echo "Deletion process completed."
