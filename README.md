# SAE Training Tool

A tool for training Sparse Autoencoders (SAEs) on transformer models to discover interpretable features.

## What is a Sparse Autoencoder?

A Sparse Autoencoder is a neural network that learns to represent data using only a few active features at a time. Think of it like a compression algorithm that finds the most important patterns in the data and represents them using a small number of "active" features.

## How it Works

1. **Input**: Takes activations from specific layers of a transformer model (like BERT)
2. **Encoding**: Compresses these activations into a sparse representation (only a few features are active)
3. **Decoding**: Reconstructs the original activations from the sparse representation
4. **Training**: Learns to minimize reconstruction error while keeping the representation sparse

## Key Metrics

- **Loss**: How well the SAE reconstructs the original activations
- **Dead Feature Percentage**: Percentage of features that are rarely used (below threshold activation rate)
- **L0 Sparsity**: Average number of active features per sample
- **Feature Absorption**: How similar features are to each other (lower is better)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd saetrain

# Install dependencies
pip install -e .
```

## Quick Start

### Option 1: Multi-GPU Training (Recommended - Fastest)
```bash
# Run multi-GPU training for fastest results
bash Train_sae_scripts_multiGPU.sh
```

**Multi-GPU Benefits:**
- 🚀 6-8x faster training with 8 GPUs
- 📊 Larger effective batch size (32 vs 4)
- 💾 Better memory efficiency
- 🔄 Distributed Data Parallel (DDP) training
- ⚡ Real-time WandB logging from all ranks

### Option 2: Single-GPU Training
```bash
# Run the complete training pipeline with automatic assessment
bash Train_sae_script.sh
```

This script includes:
- ✅ Complete training configuration
- ✅ Automatic WandB logging
- ✅ Post-training health assessment
- ✅ SAEBench metric evaluation
- ✅ Final results summary

### Option 3: Manual Training
```bash
# Train an SAE on BERT layer 6
python -m saetrain bert-base-uncased jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --max_tokens 1000000 \
    --k 192 \
    --expansion_factor 32
```

## Multi-GPU Training Setup

### Environment Variables
```bash
# Multi-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

### GPU Configuration Parameters
- **`CUDA_VISIBLE_DEVICES`**: Specify which GPUs to use (comma-separated)
- **`CUDA_LAUNCH_BLOCKING`**: Synchronous CUDA operations for debugging
- **`CUBLAS_WORKSPACE_CONFIG`**: Optimize CUBLAS memory usage
- **`PYTHONHASHSEED`**: Ensure reproducible results

### Torchrun Command
```bash
torchrun \
    --nproc_per_node=8 \
    -m saetrain \
    bert-base-uncased \
    jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --batch_size 4 \
    --k 32 \
    --num_latents 200 \
    --grad_acc_steps 8 \
    --ctx_len 512 \
    --save_dir "./test_output_torchrun" \
    --shuffle_seed 42 \
    --init_seeds 42 \
    --optimizer adam \
    --lr 0.01 \
    --save_every 500 \
    --run_name "bert_layer6_k32_latents200_torchrun" \
    --log_to_wandb true \
    --wandb_log_frequency 10 \
    --dead_percentage_threshold 0.1
```

## Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Model & Dataset** |
| `model_name` | str | - | Name of the transformer model (e.g., "bert-base-uncased") |
| `dataset_name` | str | - | Name of the dataset (e.g., "wikitext", "squad") |
| **Training Configuration** |
| `--layers` | int | 0 | Which layer(s) to train on (0-indexed) |
| `--max_tokens` | int | 1000000 | Maximum number of tokens to process |
| `--batch_size` | int | 4 | Number of samples per batch |
| `--k` | int | 192 | Number of top features to keep (sparsity parameter) |
| `--expansion_factor` | int | 32 | Multiplier for SAE size (d_sae = d_in * expansion_factor) |
| `--num_latents` | int | - | Direct SAE size (overrides expansion_factor) |
| `--grad_acc_steps` | int | 1 | Gradient accumulation steps |
| `--ctx_len` | int | 512 | Context length (sequence length) |
| **Optimization** |
| `--optimizer` | str | "adam" | Optimizer to use ("adam", "sgd") |
| `--lr` | float | 0.001 | Learning rate |
| `--weight_decay` | float | 0.0 | Weight decay |
| **Dead Feature Detection** |
| `--dead_percentage_threshold` | float | 0.0005 | Threshold for dead features (0.05% = 0.0005) |
| **Saving & Logging** |
| `--save_dir` | str | "./output" | Directory to save models |
| `--save_every` | int | 1000 | Save model every N steps |
| `--run_name` | str | - | Name for this training run |
| `--log_to_wandb` | bool | false | Whether to log to Weights & Biases |
| `--wandb_log_frequency` | int | 100 | How often to log metrics |
| **Reproducibility** |
| `--shuffle_seed` | int | 42 | Seed for dataset shuffling |
| `--init_seeds` | int | 42 | Seed for model initialization |
| **Data Processing** |
| `--data_preprocessing_num_proc` | int | 1 | Number of processes for data preprocessing |
| **Advanced** |
| `--loss_fn` | str | "mse" | Loss function ("mse", "ce", "kl", "fvu") |
| `--layer_stride` | int | 1 | Stride between layers |
| `--dead_feature_threshold` | int | 10000000 | Old dead feature threshold (deprecated) |

## Multi-GPU Training Scripts

### 1. Direct Torchrun Training (`Train_sae_scripts_multiGPU.sh`)
**Features:**
- 🚀 Direct torchrun command (like sparsify)
- 📊 8-GPU DDP training
- ⚡ Fast training with optimized parameters
- 📈 Real-time WandB logging
- 🔍 Automatic post-training assessment

**Configuration:**
```bash
Model: bert-base-uncased
Dataset: jyanimaulik/yahoo_finance_stockmarket_news
Layers: 6
Batch Size: 4 (effective: 32 with 8 GPUs)
TopK: 32
Num Latents: 200
Context Length: 512
Max Tokens: 10,000,000 (10M)
Learning Rate: 0.01
Dead Feature Threshold: 0.1 (10%)
```

### 2. Experimental Training (`Train_sae_scripts_experiment.sh`)
**Features:**
- 🔬 Experimental dead feature tracking
- 📊 Enhanced metrics collection
- 🎯 Optimized for research
- 📈 Comprehensive logging

### 3. Single-GPU Comparison (`Train_sae_scripts_experiment_single_gpu_comparison.sh`)
**Features:**
- 📊 Performance comparison
- 🔍 Detailed analysis
- 📈 Baseline metrics

## Integrated Training Script

The `Train_sae_script.sh` provides a complete training pipeline with automatic assessment:

### Features
- **Complete Configuration**: Pre-configured for BERT layer 6 training
- **Large Dataset**: Uses 500M tokens for robust training
- **Optimized Parameters**: Balanced expansion factor (8) and sparsity (k=192)
- **Automatic Assessment**: Post-training health evaluation using SAEBench standards
- **WandB Integration**: Real-time metric monitoring
- **Error Handling**: Robust run ID extraction and validation

### Configuration Details
```bash
Model: bert-base-uncased
Layer: 6
Max Tokens: 500,000,000 (500M)
Batch Size: 4
TopK: 192
Expansion Factor: 8
Context Length: 512
Learning Rate: 0.001
Dead Feature Threshold: 0.05%
```

### Output
After training, the script automatically:
1. Extracts final metrics from WandB
2. Evaluates against SAEBench thresholds
3. Provides health assessment (4/4 metrics)
4. Saves detailed results to `final_assessment.json`

### Health Metrics Evaluated
- **Loss Recovered**: ≥60-70% (reconstruction quality)
- **L0 Sparsity**: 20-200 range (feature utilization)
- **Dead Features**: ≤10-20% (feature efficiency)
- **Feature Absorption**: ≤0.25 (feature diversity)

## Examples

### Basic Training
```bash
python -m saetrain bert-base-uncased wikitext \
    --layers 6 \
    --max_tokens 1000000 \
    --k 192 \
    --expansion_factor 32
```

### Multi-GPU Training with Custom Parameters
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Run with torchrun
torchrun \
    --nproc_per_node=8 \
    -m saetrain \
    bert-base-uncased \
    jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --batch_size 4 \
    --k 32 \
    --num_latents 200 \
    --grad_acc_steps 8 \
    --ctx_len 512 \
    --save_dir "./test_output_torchrun" \
    --shuffle_seed 42 \
    --init_seeds 42 \
    --optimizer adam \
    --lr 0.01 \
    --save_every 500 \
    --run_name "bert_layer6_k32_latents200_torchrun" \
    --log_to_wandb true \
    --wandb_log_frequency 10 \
    --dead_percentage_threshold 0.1
```

### Training with Custom Parameters
```bash
python -m saetrain bert-base-uncased jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --max_tokens 500000000 \
    --batch_size 4 \
    --k 192 \
    --expansion_factor 32 \
    --grad_acc_steps 8 \
    --ctx_len 512 \
    --optimizer adam \
    --lr 0.001 \
    --save_dir "./my_sae_output" \
    --run_name "bert_layer6_large_dataset" \
    --log_to_wandb true \
    --dead_percentage_threshold 0.0005
```

### Training Multiple Layers
```bash
python -m saetrain bert-base-uncased wikitext \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --max_tokens 1000000 \
    --k 192 \
    --expansion_factor 32
```

## Understanding the Parameters

### Core Parameters
- **`k`**: Controls sparsity. Higher k = more active features = less sparse
- **`expansion_factor`**: Controls SAE size. Higher = more features to learn
- **`num_latents`**: Direct control over SAE size (overrides expansion_factor)
- **`max_tokens`**: How much data to train on. More = better but slower

### Dead Feature Detection
- **`dead_percentage_threshold`**: Features activated less than this percentage are considered "dead"
- Default 0.0005 means features must be active in at least 0.05% of tokens to be "alive"
- Higher thresholds (0.1 = 10%) are more lenient, lower thresholds are stricter

### Training Control
- **`batch_size`**: Larger = faster but more memory
- **`grad_acc_steps`**: Accumulate gradients over multiple batches (effective batch size = batch_size * grad_acc_steps)
- **`ctx_len`**: Length of text sequences. Longer = more context but more memory

### Multi-GPU Specific
- **`--nproc_per_node`**: Number of GPUs to use
- **Effective batch size**: batch_size × num_gpus × grad_acc_steps
- **DDP synchronization**: Automatic gradient synchronization across GPUs

## Output

The tool saves:
- Trained SAE models
- Training logs
- Metrics (if using WandB)
- Post-training assessment reports

## Monitoring Training

If using WandB (`--log_to_wandb true`), you can monitor:
- **Loss**: Reconstruction quality
- **Dead Feature Percentage**: How many features are unused
- **L0 Sparsity**: Average active features per sample
- **Feature Absorption**: Feature similarity (lower is better)

### Multi-GPU Monitoring
- **Distributed training**: All GPUs log to the same WandB run
- **Real-time sync**: Metrics are synchronized across all ranks
- **Performance tracking**: Training speed and GPU utilization

## Tips

1. **Start with smaller `max_tokens` for testing**
2. **Use `expansion_factor` 32 for most cases**
3. **Set `k` to 192 for good sparsity**
4. **Use `grad_acc_steps` to increase effective batch size**
5. **Monitor dead feature percentage - high values suggest the SAE is too large**
6. **For multi-GPU: Use `num_latents` for direct size control**
7. **Adjust `dead_percentage_threshold` based on your needs (0.1 for lenient, 0.0005 for strict)**
8. **Use `wandb_log_frequency` to control logging overhead**

## Performance Comparison

| Training Type | GPUs | Speed | Memory | Batch Size |
|---------------|------|-------|--------|------------|
| Single-GPU | 1 | 1x | High | 4 |
| Multi-GPU | 8 | 6-8x | Distributed | 32 (4×8) |
| Multi-GPU + Grad Acc | 8 | 6-8x | Distributed | 256 (4×8×8) |

## Troubleshooting

### Multi-GPU Issues
- **CUDA out of memory**: Reduce batch_size or use gradient accumulation
- **DDP sync issues**: Check CUDA_VISIBLE_DEVICES and torchrun parameters
- **Slow training**: Ensure all GPUs are being utilized (check nvidia-smi)

### Dead Feature Issues
- **100% dead features**: Reduce learning rate or increase dead_percentage_threshold
- **Too many dead features**: Increase k or reduce expansion_factor
- **No dead features**: Increase dead_percentage_threshold or reduce k
