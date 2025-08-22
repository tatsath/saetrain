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
    --expansion_factor 8 \
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
echo "✅ Training completed! Running post-training assessment..."

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

# Run integrated post-training assessment
python -c "
import os
import sys
import json
import wandb
from typing import Dict, Any

def get_final_metrics(run_id: str) -> Dict[str, float]:
    print(f'📊 Extracting final metrics from WandB run: {run_id}')
    
    api = wandb.Api()
    try:
        run = api.run(f'tatsatx-university-of-california-berkeley/saetrain/{run_id}')
        history = run.history()
        
        if history.empty:
            print('❌ No metrics found in the run')
            return {}
        
        final_metrics = history.iloc[-1]
        metrics = {}
        
        # FVU (Fraction of Variance Unexplained)
        fvu_key = None
        for key in final_metrics.keys():
            if 'fvu/' in key:
                fvu_key = key
                break
        
        if fvu_key:
            fvu = final_metrics[fvu_key]
            loss_recovered = (1.0 - fvu) * 100
            metrics['loss_recovered'] = loss_recovered
            print(f'   FVU: {fvu:.6f} -> Loss Recovered: {loss_recovered:.2f}%')
        
        # Dead Features Percentage
        dead_pct_key = None
        for key in final_metrics.keys():
            if 'dead_pct/' in key:
                dead_pct_key = key
                break
        
        if dead_pct_key:
            dead_pct = final_metrics[dead_pct_key] * 100
            metrics['dead_features_percent'] = dead_pct
            print(f'   Dead Features: {dead_pct:.2f}%')
        
        # L0 Sparsity
        l0_key = None
        for key in final_metrics.keys():
            if 'l0_sparsity/' in key:
                l0_key = key
                break
        
        if l0_key:
            l0_sparsity = final_metrics[l0_key]
            metrics['l0_sparsity'] = l0_sparsity
            print(f'   L0 Sparsity: {l0_sparsity:.2f}')
        
        # Feature Absorption
        absorption_key = None
        for key in final_metrics.keys():
            if 'feature_absorption/' in key:
                absorption_key = key
                break
        
        if absorption_key:
            absorption = final_metrics[absorption_key]
            metrics['feature_absorption'] = absorption
            print(f'   Feature Absorption: {absorption:.4f}')
        
        return metrics
        
    except Exception as e:
        print(f'❌ Error extracting metrics: {e}')
        print(f'   Run ID: {run_id}')
        print(f'   Project: tatsatx-university-of-california-berkeley/saetrain')
        return {}

def assess_health(metrics: Dict[str, float]) -> Dict[str, Any]:
    print('\\n🔍 Health Assessment (SAEBench Standards):')
    
    assessment = {}
    
    # 1. Loss Recovered
    if 'loss_recovered' in metrics:
        loss_recovered = metrics['loss_recovered']
        is_healthy = loss_recovered >= 60
        status = '✅ Healthy' if is_healthy else '❌ Below threshold'
        assessment['loss_recovered'] = {
            'value': loss_recovered,
            'healthy_range': '≥60-70% (SAEBench standard)',
            'is_healthy': is_healthy,
            'status': status
        }
        print(f'   Loss Recovered: {loss_recovered:.2f}% {status}')
    
    # 2. L0 Sparsity
    if 'l0_sparsity' in metrics:
        l0_sparsity = metrics['l0_sparsity']
        is_healthy = 20 <= l0_sparsity <= 200
        status = '✅ Healthy' if is_healthy else '❌ Outside range'
        assessment['l0_sparsity'] = {
            'value': l0_sparsity,
            'healthy_range': '20 ≤ L0 ≤ 200 (sweet spot: 40-120)',
            'is_healthy': is_healthy,
            'status': status
        }
        print(f'   L0 Sparsity: {l0_sparsity:.2f} {status}')
    
    # 3. Dead Features
    if 'dead_features_percent' in metrics:
        dead_features = metrics['dead_features_percent']
        is_healthy = dead_features <= 20
        status = '✅ Healthy' if is_healthy else '❌ Too many dead features'
        assessment['dead_features'] = {
            'value': dead_features,
            'healthy_range': '≤10-20% (SAEBench standard)',
            'is_healthy': is_healthy,
            'status': status
        }
        print(f'   Dead Features: {dead_features:.2f}% {status}')
    
    # 4. Feature Absorption
    if 'feature_absorption' in metrics:
        absorption = metrics['feature_absorption']
        is_healthy = absorption <= 0.25
        is_borderline = 0.25 < absorption <= 0.35
        if is_healthy:
            status = '✅ Healthy'
        elif is_borderline:
            status = '⚠️ Borderline'
        else:
            status = '❌ High absorption'
        assessment['feature_absorption'] = {
            'value': absorption,
            'healthy_range': '≤0.25 (≤0.35 borderline)',
            'is_healthy': is_healthy,
            'status': status
        }
        print(f'   Feature Absorption: {absorption:.4f} {status}')
    
    return assessment

def print_summary(assessment: Dict[str, Any]):
    print('\\n' + '='*70)
    print('📊 FINAL SAE HEALTH ASSESSMENT')
    print('='*70)
    
    healthy_metrics = sum(1 for metric in assessment.values() if metric['is_healthy'])
    total_metrics = len(assessment)
    
    print(f'🎯 OVERALL HEALTH: {healthy_metrics}/{total_metrics} metrics healthy')
    
    if healthy_metrics == total_metrics:
        print('✅ SAE is in healthy range across all metrics!')
    elif healthy_metrics >= total_metrics * 0.75:
        print('⚠️ SAE is mostly healthy with some areas for improvement')
    else:
        print('❌ SAE needs improvement in multiple areas')
    
    print('='*70)
    print('\\n📚 Methodology: SAEBench standards')
    print('🔗 Based on real training metrics from WandB')

def save_assessment(metrics: Dict[str, float], assessment: Dict[str, Any], run_id: str):
    output_data = {
        'run_id': run_id,
        'final_metrics': metrics,
        'health_assessment': assessment,
        'methodology': 'SAEBench standards'
    }
    
    with open('final_assessment.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f'\\n💾 Assessment saved to: final_assessment.json')

# Main assessment
metrics = get_final_metrics('$RUN_ID')
if metrics:
    assessment = assess_health(metrics)
    print_summary(assessment)
    save_assessment(metrics, assessment, '$RUN_ID')
else:
    print('❌ No metrics found. Cannot perform assessment.')
"

echo ""
echo "📁 Check results in: /home/nvidia/Documents/Hariom/saetrain/test_output"
echo "📊 Check WandB dashboard for detailed metrics and charts"
echo "📋 Final assessment saved to: final_assessment.json"
echo ""
echo "🔍 Key Metrics Monitored:"
echo "  - fvu/encoder.layer.6: Fraction of Variance Unexplained (loss)"
echo "  - dead_pct/encoder.layer.6: Time-based dead feature percentage"
echo "  - l0_sparsity/encoder.layer.6: Average number of active features per sample"
echo "  - feature_absorption/encoder.layer.6: Fast covariance-based absorption proxy"
echo ""
echo "📊 SAEBench Health Thresholds:"
echo "  - Loss Recovered: ≥60-70% (SAEBench standard)"
echo "  - L0 Sparsity: 20 ≤ L0 ≤ 200 (sweet spot: 40-120)"
echo "  - Dead Features: ≤10-20% (SAEBench standard)"
echo "  - Feature Absorption: ≤0.25 (≤0.35 borderline)"
