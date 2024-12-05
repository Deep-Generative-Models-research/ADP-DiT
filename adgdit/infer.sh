#!/bin/bash

# Define fixed parameters
PROMPT="90-years-old male, Advanced stage of Alzheimer’s disease, Starting from a cognitively normal brain, Includes hippocampal atrophy and ventricular enlargement"
NEGATIVE="low quality, blurry, distorted anatomy, extra artifacts, non-medical objects, unrelated symbols, human faces, missing brain regions, incorrect contrast, cartoonish, noise, grainy patches"
IMAGE_PATH="/workspace/dataset/AD/images/002_S_0295_2011-06-02_002_S_0295_2011-06-02_07_58_50.0_170.png"
DIT_WEIGHT="/workspace/log_EXP/002-dit_XL_2/checkpoints/final.pt"
LOAD_KEY="module"
OUTPUT_DIR="results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Iterate through --cfg-scale values (6 to 30)
for CFG_SCALE in {6..30}; do
    # Iterate through --infer-steps values (90 to 200 in steps of 5)
    for INFER_STEPS in $(seq 90 5 200); do
        # Execute the command
        echo "Running experiment with --cfg-scale=$CFG_SCALE and --infer-steps=$INFER_STEPS"
        python sample_t2i.py \
            --infer-mode fa \
            --prompt "$PROMPT" \
            --negative "$NEGATIVE" \
            --image-path "$IMAGE_PATH" \
            --no-enhance \
            --dit-weight "$DIT_WEIGHT" \
            --load-key "$LOAD_KEY" \
            --cfg-scale $CFG_SCALE \
            --infer-steps $INFER_STEPS

        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "Experiment with --cfg-scale=$CFG_SCALE and --infer-steps=$INFER_STEPS completed successfully."
        else
            echo "Experiment with --cfg-scale=$CFG_SCALE and --infer-steps=$INFER_STEPS failed."
        fi
    done
done

# Print completion message
echo "All experiments completed."
