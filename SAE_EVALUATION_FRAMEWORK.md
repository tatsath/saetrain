# SAE Evaluation Framework: General Guidelines & Best Practices

## 📋 Overview

This document provides a comprehensive framework for evaluating Sparse Autoencoders (SAEs) across different domains, model architectures, and datasets. It covers theoretical foundations, evaluation metrics, domain-specific considerations, and practical implementation guidelines.

## 🎯 Theoretical Foundations

### What are Sparse Autoencoders?

Sparse Autoencoders are neural networks designed to learn sparse, interpretable representations of model activations. They consist of:

1. **Encoder**: Compresses activations to sparse latent representations
2. **Decoder**: Reconstructs original activations from sparse latents
3. **Sparsity Constraint**: Ensures only a small subset of features are active

### Why Evaluate SAEs?

SAE evaluation serves multiple critical purposes:

1. **Interpretability**: Assess whether learned features are meaningful and interpretable
2. **Reconstruction Quality**: Measure how well the SAE preserves information
3. **Sparsity Validation**: Ensure the SAE maintains desired sparsity properties
4. **Generalization**: Test performance across different domains and tasks
5. **Training Health**: Identify and diagnose training issues

## 📊 Core Evaluation Metrics

### 1. Loss Recovered (Fraction of Variance Explained - FVU)

**Definition**: `(1 - MSE/Total_Variance) × 100%` - Measures how well SAE reconstructs original activations.

**Significance**: Indicates information preservation and training quality. Scale-invariant metric widely used in SAE literature.

**Healthy Ranges by Model Size**:
- **Small Models (<1B)**: 40-60%
- **Medium Models (1-10B)**: 60-80%
- **Large Models (>10B)**: 70-90%

### 2. L0 Sparsity (Average Active Features)

**Definition**: `mean(count_nonzero(activations, axis=1))` - Average number of active features per sample.

**Significance**: Ensures sparse representations for computational efficiency and interpretability. Direct measurement of actual active features.

**Optimal Ranges by Use Case**:
- **Interpretability Focus**: 10-50 features
- **Balanced Approach**: 50-150 features
- **Reconstruction Focus**: 100-300 features
- **Maximum**: 500 features (beyond this, sparsity loses meaning)

### 3. Dead Features Percentage

**Definition**: `(features_with_usage < threshold) / total_features × 100%` - Percentage of rarely or never activated features.

**Significance**: Indicates training health and capacity utilization. High dead features suggest training problems or poor initialization.

**Acceptable Ranges**:
- **Excellent**: 0-5%
- **Good**: 5-15%
- **Acceptable**: 15-25%
- **Problematic**: >25%

### 4. Feature Absorption (Decoder Weight Correlation)

**Definition**: `mean(cosine_similarity(decoder_weights))` - Average correlation between decoder weights.

**Significance**: Measures feature diversity and redundancy. High absorption indicates overlapping features, low diversity suggests better interpretability.

**Healthy Ranges**:
- **Excellent**: 0-0.15
- **Good**: 0.15-0.25
- **Acceptable**: 0.25-0.35
- **Problematic**: >0.35

## 🔄 In-Sample vs Out-of-Sample Evaluation

### In-Sample Evaluation

**Purpose**: Assess SAE performance on the training distribution
- **Metrics**: All four core metrics
- **Dataset**: Same as training data
- **Significance**: Measures learning capacity and training success

**When to Use**:
- **Training Monitoring**: During training to track progress
- **Capacity Assessment**: Determine if SAE dimension is appropriate
- **Baseline Establishment**: Set performance expectations

### Out-of-Sample (OOS) Evaluation

**Purpose**: Test generalization to unseen domains and distributions
- **Metrics**: Focus on reconstruction quality and sparsity consistency
- **Datasets**: Different domains, tasks, or distributions
- **Significance**: Measures real-world applicability

**Why OOS is Critical**:
- **Generalization Test**: Ensures SAE works beyond training data
- **Robustness Assessment**: Identifies overfitting to training distribution
- **Practical Validation**: Real-world usage involves unseen data

**OOS Dataset Categories**:

#### 1. **Domain Shift Datasets**
- **Purpose**: Test performance on different text domains
- **Examples**: News → Fiction, Academic → Social Media
- **Expected Behavior**: Moderate performance degradation (20-40% drop)

#### 2. **Task Shift Datasets**
- **Purpose**: Test performance on different NLP tasks
- **Examples**: Classification → Generation, QA → Summarization
- **Expected Behavior**: Variable performance (0-60% drop)

#### 3. **Distribution Shift Datasets**
- **Purpose**: Test performance on different data distributions
- **Examples**: Different time periods, demographics, or sources
- **Expected Behavior**: Mild performance degradation (10-30% drop)

## 🗂️ Dataset Selection Guidelines

### Training Dataset Requirements

| Aspect | Minimum | Optimal | Maximum | Notes |
|--------|---------|---------|---------|-------|
| **Samples** | 100K | 1M-10M | 100M+ | More samples = better features |
| **Tokens** | 50M | 500M-5B | 50B+ | Context length × samples |
| **Context Length** | 256 | 512-1024 | 2048 | Longer context = richer patterns |
| **Domains** | 2-3 | 5-10 | 20+ | Multiple domains for generalization |


### Evaluation Dataset Selection

#### 1. **Primary Evaluation Dataset**
- **Purpose**: Standard benchmark for comparison
- **Characteristics**: Clean, well-structured, representative
- **Examples**: WikiText, C4, OpenWebText

#### 2. **Domain-Specific Datasets**
- **Purpose**: Test domain generalization
- **Selection**: Based on target application domains
- **Examples**: News (AG News), Reviews (IMDB), Code (CodeSearchNet)

#### 3. **Task-Specific Datasets**
- **Purpose**: Test task generalization
- **Selection**: Based on target NLP tasks
- **Examples**: QA (SQuAD), Classification (GLUE), Generation (CNN/DailyMail)

## 🎯 Domain-Specific SAE Guidelines

### Domain-Specific SAE Configurations

| Domain | Model Dim | SAE Dim | Top-K | Learning Rate | Batch Size | Focus |
|--------|-----------|---------|-------|---------------|------------|-------|
| **General Language** | 4096 | 2048 (50%) | 64 | 0.001 | 32 | Cross-domain generalization |
| **Domain-Specific** | 768 | 230 (30%) | 32 | 0.005 | 64 | Domain-specific features |
| **Code Models** | 1024 | 410 (40%) | 48 | 0.0005 | 16 | Syntax preservation |
| **Multimodal** | 2048 | 1229 (60%) | 96 | 0.0001 | 8 | Cross-modal consistency |

## ⚙️ Hyperparameter Guidelines

### Model Size Hyperparameter Guidelines

| Model Size | Parameters | SAE Dim | Top-K | Learning Rate | Batch Size | Epochs | Expected Performance |
|------------|------------|---------|-------|---------------|------------|--------|---------------------|
| **Small** | <100M | 20% | 16-32 | 0.01 | 128 | 50-100 | 40-60% loss, 20-80 L0, 5-20% dead |
| **Medium** | 100M-1B | 30% | 32-64 | 0.005 | 64 | 30-50 | 50-70% loss, 40-120 L0, 3-15% dead |
| **Large** | 1B-10B | 40% | 64-128 | 0.001 | 32 | 20-30 | 60-80% loss, 60-150 L0, 2-10% dead |
| **Very Large** | >10B | 50% | 128-256 | 0.0005 | 16 | 15-25 | 70-90% loss, 80-200 L0, 1-8% dead |

### Dataset Size Adjustments

| Dataset Size | Learning Rate | Batch Size | Epochs | Regularization | Expected Impact |
|--------------|---------------|------------|--------|----------------|-----------------|
| **Small** (<1M) | ×0.5 | ×0.5 | ×2 | ×1.5 | Higher dead features (10-25%), lower reconstruction |
| **Medium** (1-10M) | ×1.0 | ×1.0 | ×1 | ×1.0 | Balanced performance |
| **Large** (>10M) | ×1.2 | ×1.5 | ×0.7 | ×0.8 | Lower dead features (2-8%), higher reconstruction |

## 📊 Evaluation Workflow

### 1. **Pre-Training Assessment**
```python
# Baseline evaluation
baseline_metrics = evaluate_baseline(model, layer, dataset)
print(f"Baseline sparsity: {baseline_metrics['natural_sparsity']}")
print(f"Baseline variance: {baseline_metrics['total_variance']}")
```

### 2. **Training Monitoring**
```python
# Every N steps
if step % evaluation_frequency == 0:
    metrics = evaluate_sae(sae, model, layer, dataset)
    log_metrics(metrics, step)
    
    # Check for issues
    if metrics['dead_features'] > 0.25:
        print("Warning: High dead features detected")
```

### 3. **Post-Training Evaluation**
```python
# Comprehensive evaluation
datasets = ['wikitext', 'squad', 'glue', 'ag_news', 'imdb']
results = {}

for dataset in datasets:
    results[dataset] = evaluate_sae(sae, model, layer, dataset)

# Generate health report
health_report = generate_health_report(results)
```

## 🎯 Metrics & Success Criteria

### Metrics & Success Criteria

| Metric | Minimum | Optimal | Model Adjustments |
|--------|---------|---------|-------------------|
| **Loss Recovered** | 40% | 70% | Small: ×0.8, Large: ×1.1 |
| **L0 Sparsity** | 20-300 | 40-120 | Small: ×0.7, Large: ×1.3 |
| **Dead Features** | ≤25% | ≤10% | Consistent across sizes |
| **Feature Absorption** | ≤35% | ≤25% | Consistent across sizes |

## 📈 Results from Trained SAEs

### Trained SAE Results Summary

| Model | Layer | SAE Dim | Top-K | Loss Rec | L0 Sparsity | Dead Feat | Absorption | Status |
|-------|-------|---------|-------|----------|-------------|-----------|------------|--------|
| **Llama 3.1 8B** | 1 | 1536 | 32 | 93.22% ✅ | 856.27 ❌ | 78.26% ❌ | 0.400 ❌ | Excellent reconstruction, poor sparsity |
| **BERT-base** | 6 | 200 | 32 | 28.82% ❌ | 94.99 ✅ | 0.00% ✅ | 0.156 ✅ | Perfect utilization, poor reconstruction |
| **Gemma 3 270M** | 6 | 200 | 64 | 0.00% ❌ | 103.37 ✅ | 11.00% ✅ | 0.114 ✅ | Good sparsity, failed reconstruction |

### Cross-Dataset Performance
| Model | WikiText | GLUE/CoLA | AG News | IMDB | Generalization |
|-------|----------|-----------|---------|------|----------------|
| **BERT-base** | 31.08% | 0.00% | 0.00% | 37.03% | ⚠️ Variable |
| **Gemma 3 270M** | 0.00% | 0.00% | 0.00% | 0.00% | ❌ Poor |

## 🔄 Continuous Improvement Framework

### 1. **Regular Evaluation Schedule**
- **Weekly**: Run comprehensive evaluation on all SAEs
- **Monthly**: Compare against benchmarks and previous versions
- **Quarterly**: Update evaluation methodology and metrics

### 2. **Iterative Improvement Process**
```python
def iterative_improvement(sae_config, evaluation_results):
    # Identify issues
    issues = analyze_issues(evaluation_results)
    
    # Generate improvements
    improvements = generate_improvements(issues)
    
    # Test improvements
    new_results = test_improvements(sae_config, improvements)
    
    # Compare and iterate
    if new_results > evaluation_results:
        return new_results
    else:
        return evaluation_results
```

### 3. **Documentation Requirements**
- **Configuration Tracking**: Document all training parameters
- **Result Archiving**: Save detailed evaluation results
- **Analysis Recording**: Document insights and recommendations
- **Comparison Tracking**: Track improvements over time

---

**This framework provides a comprehensive approach to SAE evaluation that can be applied across different domains, model sizes, and use cases. Regular updates should be made based on new research findings and practical experience.**

**Version**: 1.0  
**Last Updated**: Based on comprehensive SAE evaluation research  
**Methodology**: SAEBench with theoretical foundations
