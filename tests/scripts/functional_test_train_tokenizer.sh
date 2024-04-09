#!/bin/bash
#test to check if whether train_tokenizer.py works.

VOCAB_SIZE=4096
DATASET_NAME="delphi-suite/stories"
COLUMN="story"  # Your Hugging Face username

# Train the tokenizer
python3 scripts/train_tokenizer.py \
    --vocab-size "$VOCAB_SIZE" \
    --dataset-name "$DATASET_NAME" \
    --column "$COLUMN" \

# Check if local file exists 
TOKENIZER_MODEL_PATH="./tok${VOCAB_SIZE}.model"
if test -f TOKENIZER_MODEL_PATH; then
  echo "Tokenizer trained."
fi