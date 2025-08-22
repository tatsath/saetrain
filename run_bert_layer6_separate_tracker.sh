#!/bin/bash
# Training script for SAE on BERT layer 6 with separate dead feature activation count tracking

echo "🚀 Starting SAE Training on BERT Layer 6 (Dead Feature Activation Count Tracking)"
echo "======================================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Change to the saetrain directory
cd /home/nvidia/Documents/Hariom/saetrain

echo "📋 Training Configuration:"
echo "  Model: bert-base-uncased"
echo "  Layer: 6"
echo "  Max Tokens: 500,000,000 (500M)"
echo "  Batch Size: 4"
echo "  TopK: 192"
echo "  Expansion Factor: 32"
echo "  Context Length: 512"
echo "  Learning Rate: 0.001"
echo "  Dead Feature Percentage Threshold: 0.05%"
echo "  Dead Feature Log Frequency: 100 steps"
echo "  WandB Logging: Enabled"
echo ""

echo "⏰ Starting training..."
echo "======================================================"

# Run the training with dead feature activation count tracking
python -m saetrain \
    bert-base-uncased \
    jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --max_tokens 500000000 \
    --batch_size 4 \
    --k 192 \
    --expansion_factor 32 \
    --grad_acc_steps 8 \
    --ctx_len 512 \
    --save_dir "./test_output" \
    --shuffle_seed 42 \
    --init_seeds 42 \
    --optimizer adam \
    --lr 0.001 \
    --save_every 500 \
    --run_name "bert_layer6_separate_tracker" \
    --log_to_wandb true \
    --wandb_log_frequency 100 \
    --dead_percentage_threshold 0.0005

echo ""
echo "======================================================"
echo "✅ Training script completed!"
echo "📁 Check results in: /home/nvidia/Documents/Hariom/saetrain/test_output"
echo "📊 Check WandB dashboard for dead feature activation count metrics"
echo ""
echo "🔍 Key Metrics to Monitor in WandB:"
echo "  - dead_feature_pct/encoder.layer.6: Dead feature percentage (rate + magnitude criterion)"
echo "  - l0_sparsity/encoder.layer.6: Average number of active features per sample"
echo "  - feature_absorption/encoder.layer.6: Feature similarity (lower is better)"
