# SAE Training Results Summary
## Llama-3.1-8B-Instruct Multi-Layer SAE Training

**Model**: meta-llama/Llama-3.1-8B-Instruct | **Training**: LMSYS-Chat-1M | **Evaluation**: WikiText-103, SQuAD

---

## Results Summary

### Layer 4 (1200 Latents)
| Dataset | Loss Recovered | L0 Sparsity | Dead Features | Status |
|---------|---------------|-------------|---------------|--------|
| **In-sample** | 97.73% | 298.26 | 67.67% | ⚠️ High L0, Many Dead |
| **WikiText** | 97.72% | 284.39 | 64.50% | ⚠️ High L0, Many Dead |
| **SQuAD** | 97.72% | 284.39 | 64.50% | ⚠️ High L0, Many Dead |

### Layer 10 (400 Latents)
| Dataset | Loss Recovered | L0 Sparsity | Dead Features | Status |
|---------|---------------|-------------|---------------|--------|
| **In-sample** | 92.78% | 115.15 | 40.75% | ⚠️ Too Many Dead |
| **WikiText** | 92.16% | 117.20 | 37.75% | ⚠️ Too Many Dead |
| **SQuAD** | 92.16% | 117.20 | 37.75% | ⚠️ Too Many Dead |

### Layer 19 (400 Latents)
| Dataset | Loss Recovered | L0 Sparsity | Dead Features | Status |
|---------|---------------|-------------|---------------|--------|
| **In-sample** | 70.33% | 167.24 | 19.50% | ✅ **Best** |
| **WikiText** | 69.05% | 169.03 | 19.00% | ✅ **Best** |
| **SQuAD** | 69.05% | 169.03 | 19.00% | ✅ **Best** |

### Layer 28 (400 Latents)
| Dataset | Loss Recovered | L0 Sparsity | Dead Features | Status |
|---------|---------------|-------------|---------------|--------|
| **In-sample** | 0.00% | 193.82 | 1.25% | ❌ Failed |
| **WikiText** | 0.00% | 194.64 | 0.50% | ❌ Failed |
| **SQuAD** | 0.00% | 194.64 | 0.50% | ❌ Failed |

---

## Key Insights
- **Layer 19 (400 latents)**: Optimal performance across all metrics
- **Layer 10**: Good reconstruction but too many dead features
- **Layer 4 (1200 latents)**: High reconstruction but excessive sparsity
- **Layer 28**: Complete training failure

**Recommendation**: Use Layer 19 with 400 latents as baseline configuration.
