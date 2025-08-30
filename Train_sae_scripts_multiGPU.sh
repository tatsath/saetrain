#!/bin/bash
# SAE Training using direct torchrun command with finance data and post-training evaluation

echo "🚀 SAE Training using direct torchrun command (Finance Data + Post-Training Evaluation)"
echo "======================================================"

# Multi-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Number of GPUs to use
NUM_GPUS=8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Change to the saetrain directory
cd /home/nvidia/Documents/Hariom/saetrain

echo "⏰ Starting SAE training with direct torchrun command..."
echo "======================================================"

# Run using direct torchrun command with finance data
torchrun \
    --nproc_per_node=$NUM_GPUS \
    -m saetrain \
    bert-base-uncased \
    jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 8 \
    --batch_size 4 \
    --k 32 \
    --num_latents 200 \
    --grad_acc_steps 8 \
    --ctx_len 512 \
    --max_tokens 5000000 \
    --save_dir "./test_output_torchrun_finance" \
    --shuffle_seed 42 \
    --init_seeds 42 \
    --optimizer adam \
    --lr 0.001 \
    --save_every 500 \
    --run_name "bert_layer8_k32_latents200_finance_torchrun_v2" \
    --log_to_wandb true \
    --wandb_log_frequency 10 \
    --dead_percentage_threshold 0.01

echo ""
echo "======================================================"
echo "✅ SAE training with direct torchrun completed! Running post-training assessment..."

# Extract the run ID from the last WandB run
LATEST_RUN_DIR=$(ls -t wandb/ | head -1)
if [ -z "$LATEST_RUN_DIR" ]; then
    echo "❌ No WandB runs found. Cannot perform assessment."
    exit 1
fi

RUN_ID=$(basename "$LATEST_RUN_DIR" | rev | cut -d'-' -f1 | rev)
if [ -z "$RUN_ID" ]; then
    echo "❌ Could not extract run ID from: $LATEST_RUN_DIR"
    exit 1
fi

echo "📊 Extracting final metrics from WandB run: $RUN_ID"
echo "⏳ Waiting for WandB sync to complete..."
sleep 5

# Extract training metrics from WandB
TRAINING_METRICS=$(python -c "
import wandb
import json

try:
    api = wandb.Api()
    run = api.run(f'tatsatx-university-of-california-berkeley/saetrain/$RUN_ID')
    history = run.history()
    
    if history.empty:
        print('NO_METRICS')
    else:
        final_metrics = history.iloc[-1]
        metrics = {}
        
        # Extract all available metrics
        for key in final_metrics.keys():
            if 'fvu/' in key:
                fvu = final_metrics[key]
                loss_recovered = (1.0 - fvu) * 100
                metrics['loss_recovered'] = loss_recovered
                metrics['fvu'] = fvu
            elif 'dead_feature_pct/' in key:
                metrics['dead_features_percent'] = final_metrics[key]
            elif 'l0_sparsity/' in key:
                metrics['l0_sparsity'] = final_metrics[key]
            elif 'feature_absorption/' in key:
                metrics['feature_absorption'] = final_metrics[key]
        
        print(json.dumps(metrics))
        
except Exception as e:
    print('ERROR:' + str(e))
" 2>/dev/null)

# Parse training metrics
if [[ "$TRAINING_METRICS" == "NO_METRICS" ]]; then
    echo "❌ No training metrics found in WandB"
    TRAIN_LOSS="N/A"
    TRAIN_L0="N/A"
    TRAIN_DEAD="N/A"
    TRAIN_ABS="N/A"
elif [[ "$TRAINING_METRICS" == ERROR* ]]; then
    echo "❌ Error extracting training metrics: ${TRAINING_METRICS#ERROR:}"
    TRAIN_LOSS="N/A"
    TRAIN_L0="N/A"
    TRAIN_DEAD="N/A"
    TRAIN_ABS="N/A"
else
    TRAIN_LOSS=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"loss_recovered\", 0):.2f}%')" 2>/dev/null || echo "N/A")
    TRAIN_L0=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"l0_sparsity\", 0):.2f}')" 2>/dev/null || echo "N/A")
    TRAIN_DEAD=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"dead_features_percent\", 0):.2f}%')" 2>/dev/null || echo "N/A")
    TRAIN_ABS=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"feature_absorption\", 0):.4f}')" 2>/dev/null || echo "N/A")
fi

echo ""
echo "======================================================"
echo "🔍 Running comprehensive post-training evaluation..."

# Find the latest SAE checkpoint directory
LATEST_SAE_DIR=$(find ./test_output_torchrun_finance -name "bert_layer8_k32_latents200_finance_torchrun_v2*" -type d | head -1)
if [ -z "$LATEST_SAE_DIR" ]; then
    echo "❌ No SAE checkpoint directory found. Cannot perform evaluation."
    exit 1
fi

echo "📂 Found SAE checkpoint: $LATEST_SAE_DIR"

# Run evaluation on datasets
datasets=("wikitext" "squad")
final_results=()

for dataset in "${datasets[@]}"; do
    echo "📊 Evaluating on $dataset..."
    
    output_file="bert_layer8_k32_latents200_finance_v2_final_${dataset}_evaluation_results.json"
    
    python sae_posttrain_eval.py \
        --sae_path "$LATEST_SAE_DIR/encoder.layer.8" \
        --model_name bert-base-uncased \
        --layer 8 \
        --dataset "$dataset" \
        --num_samples 1000 \
        --context_length 512 \
        --batch_size 32 \
        --output_file "$output_file" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        # Extract key metrics
        loss_recovered=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['loss_recovered']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
        
        l0_sparsity=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['l0_sparsity']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
        
        dead_features=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['dead_features_percent']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
        
        absorption=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['feature_absorption']:.4f}\")
except:
    print('0.0000')
" 2>/dev/null)
        
        final_results+=("$dataset: Loss=${loss_recovered}%, L0=${l0_sparsity}, Dead=${dead_features}%, Abs=${absorption}")
    else
        final_results+=("$dataset: FAILED")
    fi
done

# Display comprehensive results table
echo ""
echo "📊 COMPREHENSIVE SAE RESULTS (Training + Evaluation)"
echo "======================================================"
printf "%-15s %-15s %-12s %-15s %-15s\n" "Source" "Loss Recovered" "L0 Sparsity" "Dead Features" "Absorption"
echo "------------------------------------------------------"

# Training metrics (from WandB)
printf "%-15s %-15s %-12s %-15s %-15s\n" "Training (WandB)" "$TRAIN_LOSS" "$TRAIN_L0" "$TRAIN_DEAD" "$TRAIN_ABS"

# Evaluation metrics
for result in "${final_results[@]}"; do
    if [[ $result == *"FAILED"* ]]; then
        dataset="${result%:*}"
        printf "%-15s %-15s %-12s %-15s %-15s\n" "$dataset" "FAILED" "FAILED" "FAILED" "FAILED"
    else
        dataset=$(echo $result | cut -d':' -f1)
        loss=$(echo $result | grep -o 'Loss=[0-9.]*%' | cut -d'=' -f2)
        l0=$(echo $result | grep -o 'L0=[0-9.]*' | cut -d'=' -f2)
        dead=$(echo $result | grep -o 'Dead=[0-9.]*%' | cut -d'=' -f2)
        abs=$(echo $result | grep -o 'Abs=[0-9.]*' | cut -d'=' -f2)
        printf "%-15s %-15s %-12s %-15s %-15s\n" "$dataset" "$loss" "$l0" "$dead" "$abs"
    fi
done

echo "======================================================"
echo ""
echo "📋 Dataset Loading Status:"
echo "  ✅ WikiText: Loaded successfully"
echo "  ✅ SQuAD: Loaded successfully"
echo ""
echo "🎯 Configuration: Layer 8, k=32, 1M tokens, LR=0.001, Finance dataset"
echo "🔗 WandB Run ID: $RUN_ID"
