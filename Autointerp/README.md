# 🧠 Delphi SAE Auto-Interpretation: Hyperparameters & Usage


## 🚀 Sample CLI Commands

### ⚡ **Ultra-Fast Development (2-5 minutes)**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 50000 \
  --max_latents 20 \
  --hookpoints layers.16 \
  --scorers detection \
  --filter_bos \
  --name ultra-fast-dev
```

## 🔧 Complete Hyperparameter Reference

### 🎯 **Core Model Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--model` | `meta-llama/Meta-Llama-3-8B` | **Base LLM** to analyze | None |
| `--sparse_model` | `EleutherAI/sae-llama-3-8b-32x` | **SAE/Transcoder** model path | None |
| `--hookpoints` | `[]` | **Model layers** where SAE is attached | **HIGH** - Fewer = Faster |
| `--max_latents` | `None` | **Maximum features** to analyze | **HIGH** - Lower = Faster |

### 🧠 **Explainer Model Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--explainer_model` | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | **LLM used to generate explanations** | **HIGH** - Smaller = Faster |
| `--explainer_model_max_len` | `5120` | **Maximum context length** for explainer | **MEDIUM** - Lower = Faster |
| `--explainer_provider` | `offline` | **How to run explainer** | None |
| `--explainer` | `default` | **Explanation strategy** | None |

### 📊 **Scoring Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--scorers` | `['fuzz', 'detection']` | **Quality metrics** to evaluate | **HIGH** - Fewer = Faster |
| `--num_examples_per_scorer_prompt` | `5` | **Examples per prompt** for scoring | **MEDIUM** - Lower = Faster |

### 🗃️ **Dataset & Caching Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--dataset_repo` | `EleutherAI/SmolLM2-135M-10B` | **Dataset source** for generating activations | **MEDIUM** - Smaller = Faster |
| `--dataset_split` | `train[:1%]` | **Dataset portion** to use | **HIGH** - Smaller = Much Faster |
| `--dataset_name` | `` | **Custom dataset name** | None |
| `--dataset_column` | `text` | **Column containing text** data | None |
| `--n_tokens` | `10000000` | **Total tokens** to process | **HIGH** - Lower = Much Faster |
| `--batch_size` | `32` | **Sequences per batch** | **MEDIUM** - Optimize for GPU |
| `--cache_ctx_len` | `256` | **Context length** for each sequence | **MEDIUM** - Lower = Faster |
| `--n_splits` | `5` | **Number of safetensors files** | **LOW** - Fewer = Slightly Faster |

### 🔍 **Example Construction Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--min_examples` | `200` | **Minimum examples** needed per feature | **MEDIUM** - Lower = Faster |
| `--n_examples_train` | `40` | **Training examples** for explanation | **MEDIUM** - Lower = Faster |
| `--n_examples_test` | `50` | **Testing examples** for validation | **MEDIUM** - Lower = Faster |
| `--n_non_activating` | `50` | **Negative examples** to contrast | **MEDIUM** - Lower = Faster |
| `--example_ctx_len` | `32` | **Length of each example** sequence | **MEDIUM** - Lower = Faster |
| `--center_examples` | `True` | **Center examples** on activation point | None |
| `--non_activating_source` | `random` | **Source of negative examples** | **HIGH** - FAISS = Slower |
| `--neighbours_type` | `co-occurrence` | **Type of neighbor search** | **LOW** - Different types have minimal impact |

### 🎲 **Sampling Strategy Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--n_examples_train` | `40` | **Training examples** for explanation | **MEDIUM** - Lower = Faster |
| `--n_examples_test` | `50` | **Testing examples** for validation | **MEDIUM** - Lower = Faster |
| `--n_quantiles` | `10` | **Number of activation quantiles** | **LOW** - Lower = Slightly Faster |
| `--train_type` | `quantiles` | **How to sample training examples** | None |
| `--test_type` | `quantiles` | **How to sample testing examples** | None |
| `--ratio_top` | `0.2` | **Ratio of top examples** to use | None |

### 🔧 **Technical Parameters**

| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--pipeline_num_proc` | `120` | **CPU processes** for data processing | **LOW** - Optimize for your CPU |
| `--num_gpus` | `8` | **GPU count** for model inference | **MEDIUM** - Fewer = Less overhead |
| `--seed` | `22` | **Random seed** for reproducibility | None |
| `--verbose` | `True` | **Detailed logging** output | None |
| `--filter_bos` | `False` | **Filter beginning-of-sequence tokens** | None |
| `--log_probs` | `False` | **Gather log probabilities** | **MEDIUM** - Disable = Faster |
| `--load_in_8bit` | `False` | **8-bit model loading** for memory efficiency | **MEDIUM** - Enable = Faster |
| `--hf_token` | `None` | **HuggingFace API token** | None |
| `--overwrite` | `[]` | **What to overwrite** | None |

---

## 🚀 Sample CLI Commands

### ⚡ **Ultra-Fast Development (2-5 minutes)**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 50000 \
  --max_latents 20 \
  --hookpoints layers.16 \
  --scorers detection \
  --filter_bos \
  --name ultra-fast-dev
```

### 🏃 **Fast Production (15-30 minutes)**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 2000000 \
  --max_latents 200 \
  --hookpoints layers.16 \
  --scorers detection recall \
  --filter_bos \
  --name fast-production
```

### 🎯 **Balanced Quality (1-2 hours)**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 5000000 \
  --max_latents 500 \
  --hookpoints layers.16 \
  --scorers detection recall fuzz \
  --filter_bos \
  --name balanced-quality
```

### 🏆 **Full Quality Research (3-6 hours)**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 10000000 \
  --max_latents 1000 \
  --hookpoints layers.16 \
  --scorers detection recall fuzz simulation \
  --filter_bos \
  --name full-quality-research
```

### 🔍 **Custom Dataset Example**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 1000000 \
  --max_latents 100 \
  --hookpoints layers.16 \
  --dataset_repo "jyanimaulik/yahoo_finance_stockmarket_news" \
  --dataset_split "train[:1000]" \
  --scorers detection recall \
  --filter_bos \
  --name custom-finance-dataset
```

### ⚙️ **Advanced Configuration Example**

```bash
python -m delphi \
  meta-llama/Llama-2-7b-hf \
  /path/to/your/sae \
  --n_tokens 5000000 \
  --max_latents 300 \
  --hookpoints layers.16 \
  --scorers detection recall fuzz \
  --filter_bos \
  --explainer_model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ" \
  --explainer_model_max_len 4096 \
  --min_examples 150 \
  --n_examples_train 30 \
  --n_examples_test 40 \
  --non_activating_source "random" \
  --batch_size 16 \
  --cache_ctx_len 512 \
  --name advanced-config
```

---

## 🎯 Speed Optimization Priority

### 📈 **Order of Impact (What to Lower First)**
1. **`--n_tokens`** (cache less data) - **HIGHEST IMPACT**
2. **`--max_latents`** (label fewer features) - **HIGH IMPACT**
3. **Explainer model size/quantization** - **HIGH IMPACT**
4. **Examples per feature** - **MEDIUM IMPACT**
5. **Disable FAISS** (or cache embeddings) - **MEDIUM IMPACT**
6. **Scorers** (run detection only first) - **MEDIUM IMPACT**

### 💡 **Pro Tips**
- **Start small**: Use 50K tokens and 20 features first
- **Disable FAISS**: Use `--non_activating_source random` for speed
- **Small explainer**: Use quantized models (AWQ/INT4)
- **One layer**: Analyze one layer at a time
- **Dataset slices**: Use `train[:1000]` for quick testing

---

*Run `python -m delphi --help` for complete parameter details* 🚀
