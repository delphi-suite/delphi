#!/bin/bash
#test to check if whether inference.py uploads log probabilities to Hugging Face.
#similar to generate_logprobs.sh, much smaller.

BATCH_SIZE=80
DATASET_NAME="delphi-suite/tinystories-v2-clean-tokenized"
USERNAME="transcendingvictor"  # Your Hugging Face username
TOKEN="hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # Your Hugging Face API token

# List of models
declare -a MODEL_NAMES=("delphi-suite/delphi-llama2-100k"
                        "delphi-suite/delphi-llama2-200k"
                        )

# Loop through each model and generate log probabilities
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    echo "Processing $MODEL_NAME"
    python scripts/inference.py "$MODEL_NAME" --batch-size "$BATCH_SIZE" --dataset-name "$DATASET_NAME" --username "$USERNAME" --token "$TOKEN" --test-funct
done

echo "All models processed."