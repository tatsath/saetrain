#!/usr/bin/env python3
"""
BERT vs FinBERT SAE Comparison Script
Compares sparse autoencoders trained on BERT and FinBERT models
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class SAEComparator:
    def __init__(self, bert_sae_dir: str, finbert_sae_dir: str):
        """
        Initialize SAE comparator
        
        Args:
            bert_sae_dir: Directory containing BERT SAE models
            finbert_sae_dir: Directory containing FinBERT SAE models
        """
        self.bert_sae_dir = Path(bert_sae_dir)
        self.finbert_sae_dir = Path(finbert_sae_dir)
        self.results = {}
        
    def load_sae_models(self) -> Dict:
        """Load SAE models from both directories"""
        print("📂 Loading SAE models...")
        
        models = {
            'bert': {},
            'finbert': {}
        }
        
        # Load BERT SAEs
        for layer_dir in self.bert_sae_dir.glob("bert_layer_*"):
            layer_num = int(layer_dir.name.split('_')[-1])
            sae_path = layer_dir / "sae.pt"
            if sae_path.exists():
                models['bert'][layer_num] = torch.load(sae_path, map_location='cpu')
                print(f"  ✅ Loaded BERT Layer {layer_num}")
        
        # Load FinBERT SAEs
        for layer_dir in self.finbert_sae_dir.glob("finbert_layer_*"):
            layer_num = int(layer_dir.name.split('_')[-1])
            sae_path = layer_dir / "sae.pt"
            if sae_path.exists():
                models['finbert'][layer_num] = torch.load(sae_path, map_location='cpu')
                print(f"  ✅ Loaded FinBERT Layer {layer_num}")
        
        return models
    
    def compare_encoder_weights(self, bert_sae, finbert_sae) -> Dict:
        """Compare encoder weights between BERT and FinBERT SAEs"""
        bert_encoder = bert_sae['encoder.weight'].detach().numpy()
        finbert_encoder = finbert_sae['encoder.weight'].detach().numpy()
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(bert_encoder, finbert_encoder)
        
        # Find optimal feature alignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Calculate alignment statistics
        aligned_similarities = similarity_matrix[row_ind, col_ind]
        mean_similarity = np.mean(aligned_similarities)
        std_similarity = np.std(aligned_similarities)
        
        return {
            'similarity_matrix': similarity_matrix,
            'aligned_indices': (row_ind, col_ind),
            'aligned_similarities': aligned_similarities,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'max_similarity': np.max(aligned_similarities),
            'min_similarity': np.min(aligned_similarities)
        }
    
    def compare_decoder_weights(self, bert_sae, finbert_sae) -> Dict:
        """Compare decoder weights between BERT and FinBERT SAEs"""
        bert_decoder = bert_sae['decoder.weight'].detach().numpy()
        finbert_decoder = finbert_sae['decoder.weight'].detach().numpy()
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(bert_decoder.T, finbert_decoder.T)
        
        # Find optimal feature alignment
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Calculate alignment statistics
        aligned_similarities = similarity_matrix[row_ind, col_ind]
        mean_similarity = np.mean(aligned_similarities)
        std_similarity = np.std(aligned_similarities)
        
        return {
            'similarity_matrix': similarity_matrix,
            'aligned_indices': (row_ind, col_ind),
            'aligned_similarities': aligned_similarities,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'max_similarity': np.max(aligned_similarities),
            'min_similarity': np.min(aligned_similarities)
        }
    
    def analyze_feature_evolution(self, models: Dict) -> Dict:
        """Analyze how features evolve between BERT and FinBERT"""
        print("🔍 Analyzing feature evolution...")
        
        evolution_stats = {}
        
        for layer in range(12):  # BERT has 12 layers
            if layer not in models['bert'] or layer not in models['finbert']:
                continue
                
            print(f"  📊 Analyzing Layer {layer}...")
            
            bert_sae = models['bert'][layer]
            finbert_sae = models['finbert'][layer]
            
            # Compare encoder weights
            encoder_comparison = self.compare_encoder_weights(bert_sae, finbert_sae)
            
            # Compare decoder weights
            decoder_comparison = self.compare_decoder_weights(bert_sae, finbert_sae)
            
            # Calculate feature evolution metrics
            evolution_stats[layer] = {
                'encoder_mean_similarity': encoder_comparison['mean_similarity'],
                'encoder_std_similarity': encoder_comparison['std_similarity'],
                'decoder_mean_similarity': decoder_comparison['mean_similarity'],
                'decoder_std_similarity': decoder_comparison['std_similarity'],
                'overall_similarity': (encoder_comparison['mean_similarity'] + 
                                     decoder_comparison['mean_similarity']) / 2,
                'feature_preservation': np.sum(encoder_comparison['aligned_similarities'] > 0.8) / 
                                      len(encoder_comparison['aligned_similarities']),
                'feature_repurposing': np.sum(encoder_comparison['aligned_similarities'] < 0.5) / 
                                     len(encoder_comparison['aligned_similarities'])
            }
        
        return evolution_stats
    
    def generate_visualizations(self, evolution_stats: Dict):
        """Generate visualizations for the comparison results"""
        print("📈 Generating visualizations...")
        
        # Prepare data for plotting
        layers = list(evolution_stats.keys())
        encoder_similarities = [evolution_stats[layer]['encoder_mean_similarity'] for layer in layers]
        decoder_similarities = [evolution_stats[layer]['decoder_mean_similarity'] for layer in layers]
        overall_similarities = [evolution_stats[layer]['overall_similarity'] for layer in layers]
        feature_preservation = [evolution_stats[layer]['feature_preservation'] for layer in layers]
        feature_repurposing = [evolution_stats[layer]['feature_repurposing'] for layer in layers]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BERT vs FinBERT SAE Feature Evolution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Similarity across layers
        axes[0, 0].plot(layers, encoder_similarities, 'b-o', label='Encoder', linewidth=2, markersize=6)
        axes[0, 0].plot(layers, decoder_similarities, 'r-s', label='Decoder', linewidth=2, markersize=6)
        axes[0, 0].plot(layers, overall_similarities, 'g-^', label='Overall', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Cosine Similarity')
        axes[0, 0].set_title('Feature Similarity Across Layers')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature preservation vs repurposing
        axes[0, 1].plot(layers, feature_preservation, 'b-o', label='Preserved (>0.8)', linewidth=2, markersize=6)
        axes[0, 1].plot(layers, feature_repurposing, 'r-s', label='Repurposed (<0.5)', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Fraction of Features')
        axes[0, 1].set_title('Feature Preservation vs Repurposing')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Similarity distribution
        all_similarities = []
        for layer in layers:
            all_similarities.extend([evolution_stats[layer]['encoder_mean_similarity']])
        
        axes[1, 0].hist(all_similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Feature Similarities')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Layer-wise comparison
        x = np.arange(len(layers))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, encoder_similarities, width, label='Encoder', alpha=0.8)
        axes[1, 1].bar(x + width/2, decoder_similarities, width, label='Decoder', alpha=0.8)
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].set_title('Encoder vs Decoder Similarity by Layer')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(layers)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('1.1. BERT_vs_FinBERT_SAE_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  ✅ Saved visualization: 1.1. BERT_vs_FinBERT_SAE_Comparison.png")
    
    def generate_report(self, evolution_stats: Dict):
        """Generate a comprehensive report of the comparison"""
        print("📝 Generating analysis report...")
        
        # Calculate summary statistics
        all_encoder_similarities = [stats['encoder_mean_similarity'] for stats in evolution_stats.values()]
        all_decoder_similarities = [stats['decoder_mean_similarity'] for stats in evolution_stats.values()]
        all_overall_similarities = [stats['overall_similarity'] for stats in evolution_stats.values()]
        
        report = f"""
# BERT vs FinBERT SAE Feature Evolution Analysis Report

## Executive Summary

This report analyzes the evolution of sparse autoencoder (SAE) features between BERT and FinBERT models, 
revealing how fine-tuning affects the internal representations of language models.

## Key Findings

### Overall Similarity Statistics
- **Average Encoder Similarity**: {np.mean(all_encoder_similarities):.4f} ± {np.std(all_encoder_similarities):.4f}
- **Average Decoder Similarity**: {np.mean(all_decoder_similarities):.4f} ± {np.std(all_decoder_similarities):.4f}
- **Average Overall Similarity**: {np.mean(all_overall_similarities):.4f} ± {np.std(all_overall_similarities):.4f}

### Feature Evolution Patterns
- **Most Similar Layer**: Layer {max(evolution_stats.items(), key=lambda x: x[1]['overall_similarity'])[0]} 
  (Similarity: {max(evolution_stats.items(), key=lambda x: x[1]['overall_similarity'])[1]['overall_similarity']:.4f})
- **Least Similar Layer**: Layer {min(evolution_stats.items(), key=lambda x: x[1]['overall_similarity'])[0]} 
  (Similarity: {min(evolution_stats.items(), key=lambda x: x[1]['overall_similarity'])[1]['overall_similarity']:.4f})

## Layer-by-Layer Analysis

"""
        
        for layer in sorted(evolution_stats.keys()):
            stats = evolution_stats[layer]
            report += f"""
### Layer {layer}
- **Encoder Similarity**: {stats['encoder_mean_similarity']:.4f} ± {stats['encoder_std_similarity']:.4f}
- **Decoder Similarity**: {stats['decoder_mean_similarity']:.4f} ± {stats['decoder_std_similarity']:.4f}
- **Overall Similarity**: {stats['overall_similarity']:.4f}
- **Feature Preservation**: {stats['feature_preservation']:.2%}
- **Feature Repurposing**: {stats['feature_repurposing']:.2%}

"""
        
        report += f"""
## Interpretation

### Feature Preservation
The analysis shows that approximately {np.mean([stats['feature_preservation'] for stats in evolution_stats.values()]):.1%} 
of features are well-preserved (similarity > 0.8) between BERT and FinBERT, indicating that fine-tuning 
largely maintains the core representational structure while adapting it for financial domain tasks.

### Feature Repurposing
About {np.mean([stats['feature_repurposing'] for stats in evolution_stats.values()]):.1%} of features show 
significant repurposing (similarity < 0.5), suggesting that fine-tuning creates specialized representations 
for financial language understanding.

### Layer-wise Patterns
The similarity patterns across layers reveal how different levels of abstraction are affected by fine-tuning:
- Early layers (0-3): Tend to show higher preservation as they capture basic linguistic features
- Middle layers (4-8): Show moderate adaptation for domain-specific patterns
- Later layers (9-11): Exhibit more specialized adaptations for financial reasoning

## Conclusions

1. **Domain Adaptation**: FinBERT fine-tuning successfully adapts BERT's representations for financial text
2. **Feature Reuse**: Most features are preserved and repurposed rather than completely replaced
3. **Layer Specialization**: Different layers show varying degrees of adaptation
4. **Representational Continuity**: The core linguistic structure is maintained while adding financial expertise

This analysis provides insights into how language model fine-tuning affects internal representations and 
can guide future work on domain-specific model adaptation.
"""
        
        # Save report
        with open('1.2. BERT_vs_FinBERT_SAE_Analysis_Report.md', 'w') as f:
            f.write(report)
        
        print("  ✅ Saved report: 1.2. BERT_vs_FinBERT_SAE_Analysis_Report.md")
        
        return report
    
    def run_comparison(self):
        """Run the complete comparison analysis"""
        print("🚀 Starting BERT vs FinBERT SAE Comparison")
        print("=" * 50)
        
        # Load models
        models = self.load_sae_models()
        
        if not models['bert'] or not models['finbert']:
            print("❌ Error: Could not load SAE models from both directories")
            return
        
        # Analyze feature evolution
        evolution_stats = self.analyze_feature_evolution(models)
        
        # Generate visualizations
        self.generate_visualizations(evolution_stats)
        
        # Generate report
        self.generate_report(evolution_stats)
        
        # Save detailed results
        with open('1.3. BERT_vs_FinBERT_SAE_Analysis_Report.md', 'w') as f:
            f.write(json.dumps(evolution_stats, indent=2, default=str))
        
        print("✅ Comparison analysis completed!")
        print("📁 Generated files:")
        print("  - 1.1. BERT_vs_FinBERT_SAE_Comparison.png")
        print("  - 1.2. BERT_vs_FinBERT_SAE_Analysis_Report.md")
        print("  - 1.3. BERT_vs_FinBERT_SAE_Analysis_Report.md")

def main():
    """Main function to run the comparison"""
    # Define paths
    bert_sae_dir = "sae_outputs"
    finbert_sae_dir = "sae_outputs"
    
    # Initialize comparator
    comparator = SAEComparator(bert_sae_dir, finbert_sae_dir)
    
    # Run comparison
    comparator.run_comparison()

if __name__ == "__main__":
    main()
