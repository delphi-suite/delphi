#!/bin/bash

# Define the dataset split
DATASET_SPLIT="validation"  # Change this to your desired dataset split

# Define the batch size
BATCH_SIZE=80  # Change this if you want to use a different batch size

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
    python inference.py "$MODEL_NAME" "$DATASET_SPLIT" --batch_size "$BATCH_SIZE"
done

echo "All models processed."
