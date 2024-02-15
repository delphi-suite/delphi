#!/bin/bash

# Define the batch size
BATCH_SIZE=80  # This worked well in my CPU, but 200 was too much
DATASET_NAME="delphi-suite/tinystories-v2-clean-tokenized"
TOKEN="hf_aaaaaaaaaaaaaaaaaaaaaaaaa"  # your Hugging Face API token


# List of models
declare -a MODEL_NAMES=("delphi-suite/delphi-llama2-100k"
                        "delphi-suite/delphi-llama2-200k"
                        "delphi-suite/delphi-llama2-400k"
                        "delphi-suite/delphi-llama2-800k"
                        "delphi-suite/delphi-llama2-1.6m"
                        "delphi-suite/delphi-llama2-3.2m"
                        "delphi-suite/delphi-llama2-6.4m"
                        "delphi-suite/delphi-llama2-12.8m"
                        "delphi-suite/delphi-llama2-25.6m")

# Loop through each model and generate log probabilities
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    echo "Processing $MODEL_NAME"
    python scripts/inference_delete.py "$MODEL_NAME" --batch_size "$BATCH_SIZE" --token "$TOKEN"
done

echo "All models processed."
