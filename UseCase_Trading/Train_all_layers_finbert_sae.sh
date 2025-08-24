#!/bin/bash

# FinBERT SAE Training Script for All Layers
# This script trains sparse autoencoders on all layers of FinBERT

set -e

echo "🚀 Starting FinBERT SAE Training for All Layers"
echo "================================================"

# Set environment variables for reproducibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Training parameters
MODEL_NAME="ProsusAI/finbert"
DATASET_NAME="jyanimaulik/yahoo_finance_stockmarket_news"
NUM_TOKENS=10000000  # Increased for multi-layer training
EXPANSION_FACTOR=8
NUM_LATENTS=24576
K=192
DEAD_PERCENTAGE_THRESHOLD=5.0

# Output directory
OUTPUT_DIR="sae_outputs"
mkdir -p $OUTPUT_DIR

echo "📊 Training Parameters:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Tokens: $NUM_TOKENS"
echo "  Expansion Factor: $EXPANSION_FACTOR"
echo "  Num Latents: $NUM_LATENTS"
echo "  Top-K: $K"
echo "  Dead Feature Threshold: $DEAD_PERCENTAGE_THRESHOLD%"
echo "  Output: $OUTPUT_DIR"
echo ""

# Train SAE on all layers (0-11 for FinBERT)
for layer in {0..11}; do
    echo "🔄 Training SAE for Layer $layer"
    echo "--------------------------------"
    
    python -m saetrain \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --layer $layer \
        --num_tokens $NUM_TOKENS \
        --expansion_factor $EXPANSION_FACTOR \
        --num_latents $NUM_LATENTS \
        --k $K \
        --dead_percentage_threshold $DEAD_PERCENTAGE_THRESHOLD \
        --wandb_project "finbert-sae-all-layers" \
        --wandb_name "finbert-layer-$layer" \
        --wandb_log_frequency 100 \
        --save_dir "$OUTPUT_DIR/finbert_layer_$layer" \
        --device cuda \
        --dtype bfloat16 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --num_epochs 10 \
        --save_frequency 1000 \
        --log_frequency 100
    
    echo "✅ Completed training for Layer $layer"
    echo ""
done

echo "🎉 All FinBERT layers training completed!"
echo "📁 Output saved in: $OUTPUT_DIR"
echo "📊 Check WandB for detailed metrics"
