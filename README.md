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
- **Dead Feature Percentage**: Percentage of features that are rarely used (below 0.05% activation rate)
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

```bash
# Train an SAE on BERT layer 6
python -m saetrain bert-base-uncased jyanimaulik/yahoo_finance_stockmarket_news \
    --layers 6 \
    --max_tokens 1000000 \
    --k 192 \
    --expansion_factor 32
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

## Examples

### Basic Training
```bash
python -m saetrain bert-base-uncased wikitext \
    --layers 6 \
    --max_tokens 1000000 \
    --k 192 \
    --expansion_factor 32
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
- **`max_tokens`**: How much data to train on. More = better but slower

### Dead Feature Detection
- **`dead_percentage_threshold`**: Features activated less than this percentage are considered "dead"
- Default 0.0005 means features must be active in at least 0.05% of tokens to be "alive"

### Training Control
- **`batch_size`**: Larger = faster but more memory
- **`grad_acc_steps`**: Accumulate gradients over multiple batches (effective batch size = batch_size * grad_acc_steps)
- **`ctx_len`**: Length of text sequences. Longer = more context but more memory

## Output

The tool saves:
- Trained SAE models
- Training logs
- Metrics (if using WandB)

## Monitoring Training

If using WandB (`--log_to_wandb true`), you can monitor:
- **Loss**: Reconstruction quality
- **Dead Feature Percentage**: How many features are unused
- **L0 Sparsity**: Average active features per sample
- **Feature Absorption**: Feature similarity (lower is better)

## Tips

1. Start with smaller `max_tokens` for testing
2. Use `expansion_factor` 32 for most cases
3. Set `k` to 192 for good sparsity
4. Use `grad_acc_steps` to increase effective batch size
5. Monitor dead feature percentage - high values suggest the SAE is too large
