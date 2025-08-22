#!/usr/bin/env python3
"""
SAE Evaluation Script using Real Model Activations
Follows both SAEBench and Sparsify methodologies with real data
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SAE Evaluation with Real Model Activations")
    
    # SAE checkpoint
    parser.add_argument("--sae_path", type=str, required=True,
                       help="Path to SAE checkpoint directory")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="HuggingFace model name to use for activations")
    parser.add_argument("--layer", type=int, default=12,
                       help="Model layer to extract activations from")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="HuggingFace dataset name (e.g., 'wikitext', 'squad', 'glue')")
    parser.add_argument("--dataset_column", type=str, default="text",
                       help="Dataset column containing text")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of text samples to use")
    parser.add_argument("--context_length", type=int, default=128,
                       help="Context length for tokenization")
    parser.add_argument("--max_chars_per_sample", type=int, default=1000,
                       help="Maximum characters per text sample")
    
    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Output configuration
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup device for computation"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"🔧 Using device: {device}")
    return device

def load_sae_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Load SAE checkpoint and return configuration and weights"""
    
    print(f"📂 Loading SAE from: {checkpoint_path}")
    
    # Load configuration
    config_path = os.path.join(checkpoint_path, "cfg.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load weights - try different possible file names
    possible_weights_paths = [
        os.path.join(checkpoint_path, "sae_weights.safetensors"),
        os.path.join(checkpoint_path, "sae.safetensors"),
        os.path.join(checkpoint_path, "weights.safetensors")
    ]
    
    weights_path = None
    for path in possible_weights_paths:
        if os.path.exists(path):
            weights_path = path
            break
    
    if weights_path is None:
        raise FileNotFoundError(f"Weights file not found. Tried: {possible_weights_paths}")
    
    print(f"📁 Loading weights from: {weights_path}")
    
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        # Try different possible tensor names
        tensor_names = list(f.keys())
        print(f"📋 Available tensors: {tensor_names}")
        
        # Map tensor names to expected keys
        weight_mapping = {}
        for name in tensor_names:
            if "encoder" in name.lower() or "enc" in name.lower():
                if "weight" in name.lower() or "W" in name:
                    weight_mapping["W_enc"] = name
                elif "bias" in name.lower() or "b" in name:
                    weight_mapping["b_enc"] = name
            elif "decoder" in name.lower() or "dec" in name.lower():
                if "weight" in name.lower() or "W" in name:
                    weight_mapping["W_dec"] = name
                elif "bias" in name.lower() or "b" in name:
                    weight_mapping["b_dec"] = name
            else:
                # Fallback mapping
                if "weight" in name.lower() or "W" in name:
                    if "enc" in name.lower():
                        weight_mapping["W_enc"] = name
                    elif "dec" in name.lower():
                        weight_mapping["W_dec"] = name
                    else:
                        # Assume first weight is encoder, second is decoder
                        if "W_enc" not in weight_mapping:
                            weight_mapping["W_enc"] = name
                        else:
                            weight_mapping["W_dec"] = name
                elif "bias" in name.lower() or "b" in name:
                    if "enc" in name.lower():
                        weight_mapping["b_enc"] = name
                    elif "dec" in name.lower():
                        weight_mapping["b_dec"] = name
                    else:
                        # Assume first bias is encoder, second is decoder
                        if "b_enc" not in weight_mapping:
                            weight_mapping["b_enc"] = name
                        else:
                            weight_mapping["b_dec"] = name
        
        weights = {}
        for expected_key, actual_name in weight_mapping.items():
            weights[expected_key] = f.get_tensor(actual_name).to(device)
            print(f"   {expected_key}: {actual_name} -> {weights[expected_key].shape}")
    
    # Calculate d_sae from expansion factor if not present
    if 'd_sae' not in config and 'expansion_factor' in config:
        config['d_sae'] = config['d_in'] * config['expansion_factor']
    elif 'd_sae' not in config:
        # Fallback: calculate from weights
        config['d_sae'] = weights['W_enc'].shape[0]
    
    print(f"✅ SAE loaded successfully")
    print(f"   Input dimension (d_in): {config['d_in']}")
    print(f"   SAE dimension (d_sae): {config['d_sae']}")
    print(f"   Expansion factor: {config['d_sae'] / config['d_in']:.1f}")
    
    return config, weights

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer"""
    print(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - handle different model types
    if "bert" in model_name.lower():
        from transformers import BertModel
        model = BertModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"✅ Model loaded successfully")
    
    return model, tokenizer

def load_dataset_samples(dataset_name: str, column_name: str, num_samples: int, 
                        max_chars: int, tokenizer, verbose: bool = False) -> List[str]:
    """Load text samples from dataset"""
    print(f"📊 Loading dataset: {dataset_name}")
    
    # Try different dataset configurations (tested and working)
    # Prioritize working configurations over the original dataset_name
    dataset_configs = [
        # If the requested dataset is squad, try squad/plain_text first
        ("squad", "plain_text", False) if dataset_name == "squad" else None,
        ("squad", None, False) if dataset_name == "squad" else None,
        # If the requested dataset is wikitext, try wikitext configs first
        ("wikitext", "wikitext-103-raw-v1", False) if dataset_name == "wikitext" else None,
        ("wikitext", "wikitext-103-v1", False) if dataset_name == "wikitext" else None,
        # Original attempts
        (dataset_name, "train", True),  # Original with streaming
        (dataset_name, "train", False),  # Original without streaming
        # Fallback to working datasets
        ("wikitext", "wikitext-103-raw-v1", False),  # Primary fallback
        ("wikitext", "wikitext-103-v1", False),  # Alternative wikitext
        ("wikitext", "wikitext-2-raw-v1", False),  # Smaller wikitext
        ("wikitext", "wikitext-2-v1", False),  # Alternative smaller wikitext
        ("squad", "plain_text", False),  # SQuAD with correct config
        ("squad", None, False),  # SQuAD without config
        ("glue", "cola", False),  # GLUE CoLA
        ("glue", "sst2", False),  # GLUE SST-2
        ("imdb", None, False),  # IMDB reviews
        ("ag_news", None, False),  # AG News
    ]
    # Remove None entries
    dataset_configs = [config for config in dataset_configs if config is not None]
    
    for config_name, config_subset, use_streaming in dataset_configs:
        try:
            if config_subset:
                dataset = load_dataset(config_name, config_subset, split="train", streaming=use_streaming)
            else:
                dataset = load_dataset(config_name, split="train", streaming=use_streaming)
            
            print(f"✅ Successfully loaded: {config_name}")
            break
        except Exception as e:
            print(f"❌ Failed to load {config_name}{'/' + config_subset if config_subset else ''}: {e}")
            # Try with plain_text config for squad
            if config_name == "squad" and "BuilderConfig 'train' not found" in str(e):
                try:
                    dataset = load_dataset("squad", "plain_text", split="train", streaming=use_streaming)
                    print(f"✅ Successfully loaded: squad/plain_text")
                    break
                except Exception as e2:
                    print(f"❌ Failed to load squad/plain_text: {e2}")
            continue
    else:
        print("❌ All dataset attempts failed, using fallback samples...")
        return generate_fallback_samples(num_samples, max_chars)
    
    samples = []
    total_chars = 0
    
    try:
        for i, row in enumerate(dataset):
            if i >= num_samples * 2:  # Get extra samples in case some are too short
                break
                
            # Handle different dataset column structures
            text = ""
            if column_name in row:
                text = row.get(column_name, "")
            elif "context" in row:  # For SQuAD dataset
                text = row.get("context", "")
            elif "text" in row:  # For wikitext dataset
                text = row.get("text", "")
            elif "sentence" in row:  # For GLUE CoLA dataset
                text = row.get("sentence", "")
            
            if isinstance(text, str) and len(text) > 50 and len(text) <= max_chars:
                samples.append(text)
                total_chars += len(text)
                
                if len(samples) >= num_samples:
                    break
        
        if len(samples) < num_samples:
            print(f"⚠️ Only got {len(samples)} samples, generating fallback samples...")
            fallback_samples = generate_fallback_samples(num_samples - len(samples), max_chars)
            samples.extend(fallback_samples)
        
        print(f"✅ Loaded {len(samples)} text samples ({total_chars} total characters)")
        return samples[:num_samples]
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        print("🔄 Falling back to default text samples...")
        return generate_fallback_samples(num_samples, max_chars)

def generate_fallback_samples(num_samples: int, max_chars: int) -> List[str]:
    """Generate fallback text samples if dataset loading fails"""
    fallback_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample text for evaluation.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Transformers have revolutionized the field of natural language processing.",
        "Sparse autoencoders are used to learn efficient representations of data.",
        "The attention mechanism allows models to focus on relevant parts of input.",
        "Language models can generate human-like text through training on large datasets.",
        "Computer vision tasks include image classification and object detection.",
        "Reinforcement learning involves training agents through trial and error."
    ]
    
    samples = []
    for i in range(num_samples):
        # Repeat and modify fallback texts
        base_text = fallback_texts[i % len(fallback_texts)]
        modified_text = f"{base_text} Sample {i+1}. " * (max_chars // len(base_text) + 1)
        samples.append(modified_text[:max_chars])
    
    return samples

def tokenize_samples(samples: List[str], tokenizer, context_length: int, 
                    device: torch.device, verbose: bool = False) -> torch.Tensor:
    """Tokenize text samples"""
    print(f"🔤 Tokenizing {len(samples)} samples with context length {context_length}")
    
    all_tokens = []
    
    for i, text in enumerate(tqdm(samples, desc="Tokenizing", disable=not verbose)):
        # Tokenize the text
        tokens = tokenizer(
            text,
            max_length=context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        all_tokens.append(tokens["input_ids"])
    
    # Stack all tokenized samples
    tokens_tensor = torch.cat(all_tokens, dim=0).to(device)
    
    print(f"✅ Tokenized to shape: {tokens_tensor.shape}")
    return tokens_tensor

def extract_model_activations(model, tokens: torch.Tensor, layer: int, 
                            device: torch.device, batch_size: int, verbose: bool = False) -> torch.Tensor:
    """Extract model activations from specified layer"""
    print(f"🧠 Extracting activations from layer {layer}")
    
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Extracting activations", disable=not verbose):
            batch_tokens = tokens[i:i + batch_size]
            
            # Get activations using hooks
            activations = None
            
            def hook_fn(module, input, output):
                nonlocal activations
                if isinstance(output, tuple):
                    activations = output[0].detach()
                else:
                    activations = output.detach()
            
            # Register hook on the target layer
            if hasattr(model, 'transformer'):
                # For models like GPT-2, DialoGPT
                target_module = model.transformer.h[layer]
            elif hasattr(model, 'model'):
                # For models like Gemma, Llama
                target_module = model.model.layers[layer]
            elif hasattr(model, 'encoder'):
                # For BERT models
                target_module = model.encoder.layer[layer]
            else:
                raise ValueError(f"Unknown model architecture: {type(model)}")
            
            handle = target_module.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = model(batch_tokens)
            
            handle.remove()
            
            if activations is None:
                raise ValueError("Failed to extract activations")
            
            # Ensure activations are on the correct device
            activations = activations.to(device)
            
            # Flatten batch and sequence dimensions
            batch_size_actual, seq_len, hidden_size = activations.shape
            activations_flat = activations.view(-1, hidden_size)
            
            all_activations.append(activations_flat)
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)
    
    print(f"✅ Extracted activations shape: {all_activations.shape}")
    return all_activations

def encode_sae(activations: torch.Tensor, weights: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode activations using the SAE (following Sparsify's approach)"""
    W_enc = weights["W_enc"]  # Shape: (d_sae, d_in) - already transposed
    b_enc = weights["b_enc"]  # (d_sae,)
    
    # Apply encoder: activations @ W_enc + b_enc
    # activations: (batch_size, d_in)
    # W_enc: (d_sae, d_in) - already in correct format for F.linear
    # Result: (batch_size, d_sae)
    pre_acts = F.linear(activations, W_enc, b_enc)  # W_enc is already (d_sae, d_in)
    
    # Apply ReLU activation
    acts = F.relu(pre_acts)
    
    # Get top-k activations (following Sparsify's TopK approach)
    k = min(200, acts.shape[1])  # Use k=200 or all features if fewer
    top_acts, top_indices = acts.topk(k, dim=1, sorted=False)
    
    return top_acts, top_indices, pre_acts

def decode_sae(top_acts: torch.Tensor, top_indices: torch.Tensor, weights: Dict) -> torch.Tensor:
    """Decode encoded activations back to original space (following Sparsify's approach)"""
    W_dec = weights["W_dec"]  # (d_sae, d_in) for SAELens
    b_dec = weights["b_dec"]  # (d_in,)
    
    # Create sparse tensor with only top-k activations
    batch_size, k = top_acts.shape
    d_sae = W_dec.shape[0]  # First dimension is d_sae
    
    # Scatter the top activations back to full dimension
    sparse_acts = torch.zeros(batch_size, d_sae, device=top_acts.device, dtype=top_acts.dtype)
    sparse_acts.scatter_(1, top_indices, top_acts)
    
    # Apply decoder: sparse_acts @ W_dec + b_dec
    # sparse_acts: (batch_size, d_sae)
    # W_dec: (d_sae, d_in)
    # For F.linear, we need weight to be (out_features, in_features) = (d_in, d_sae)
    # So we need W_dec.T which is (d_in, d_sae)
    # Result: (batch_size, d_in)
    decoded = F.linear(sparse_acts, W_dec.T, b_dec)
    
    return decoded

def calculate_fvu_saebench(activations: torch.Tensor, weights: Dict, verbose: bool = False) -> float:
    """
    Calculate FVU (Fraction of Variance Unexplained) following SAEBench methodology
    Uses actual model activations and proper FVU calculation
    """
    if verbose:
        print(f"🔍 FVU Debug - Input activations shape: {activations.shape}")
        print(f"🔍 FVU Debug - Input activations range: [{activations.min():.4f}, {activations.max():.4f}]")
        print(f"🔍 FVU Debug - Input activations mean: {activations.mean():.4f}")
        print(f"🔍 FVU Debug - Input activations std: {activations.std():.4f}")
    
    # Encode and decode
    top_acts, top_indices, pre_acts = encode_sae(activations, weights)
    
    if verbose:
        print(f"🔍 FVU Debug - Pre-acts shape: {pre_acts.shape}")
        print(f"🔍 FVU Debug - Pre-acts range: [{pre_acts.min():.4f}, {pre_acts.max():.4f}]")
        print(f"🔍 FVU Debug - Top acts shape: {top_acts.shape}")
        print(f"🔍 FVU Debug - Top acts range: [{top_acts.min():.4f}, {top_acts.max():.4f}]")
        print(f"🔍 FVU Debug - Number of active features (>0): {(F.relu(pre_acts) > 0).sum().item()}")
    
    sae_out = decode_sae(top_acts, top_indices, weights)
    
    if verbose:
        print(f"🔍 FVU Debug - SAE output shape: {sae_out.shape}")
        print(f"🔍 FVU Debug - SAE output range: [{sae_out.min():.4f}, {sae_out.max():.4f}]")
        print(f"🔍 FVU Debug - SAE output mean: {sae_out.mean():.4f}")
        print(f"🔍 FVU Debug - SAE output std: {sae_out.std():.4f}")
    
    # Calculate total variance (denominator) - SAEBench approach
    activations_centered = activations - activations.mean(0)
    total_variance = activations_centered.pow(2).sum()
    
    # Calculate reconstruction error (numerator)
    reconstruction_error = activations - sae_out
    l2_loss = reconstruction_error.pow(2).sum()
    
    if verbose:
        print(f"🔍 FVU Debug - Total variance: {total_variance:.4f}")
        print(f"🔍 FVU Debug - L2 loss: {l2_loss:.4f}")
        print(f"🔍 FVU Debug - Reconstruction error mean: {reconstruction_error.mean():.4f}")
        print(f"🔍 FVU Debug - Reconstruction error std: {reconstruction_error.std():.4f}")
    
    # FVU = l2_loss / total_variance
    fvu = l2_loss / total_variance
    
    if verbose:
        print(f"🔍 FVU Debug - Raw FVU: {fvu:.6f}")
    
    # Loss recovered = (1 - FVU) * 100
    loss_recovered = (1.0 - fvu.item()) * 100
    
    if verbose:
        print(f"🔍 FVU Debug - Loss recovered: {loss_recovered:.6f}%")
    
    return max(0.0, min(100.0, loss_recovered))

def calculate_l0_sparsity_saebench(activations: torch.Tensor, weights: Dict) -> float:
    """
    Calculate L0 Sparsity following SAEBench methodology
    Uses actual model activations and counts non-zero features
    """
    # Encode to get activations
    _, _, pre_acts = encode_sae(activations, weights)
    acts = F.relu(pre_acts)
    
    # Count non-zero features per sample (L0 sparsity)
    active_features = (acts > 0).sum(dim=1).float()
    
    # Return average L0 sparsity
    return active_features.mean().item()

def calculate_dead_features_saebench(activations: torch.Tensor, weights: Dict, verbose: bool = False) -> float:
    """
    Calculate dead features following SAEBench methodology
    Features that are never activated across the dataset
    """
    # Encode to get activations
    _, _, pre_acts = encode_sae(activations, weights)
    acts = F.relu(pre_acts)
    
    if verbose:
        print(f"🔍 Dead Features Debug - Pre-acts shape: {pre_acts.shape}")
        print(f"🔍 Dead Features Debug - Acts shape: {acts.shape}")
        print(f"🔍 Dead Features Debug - Acts range: [{acts.min():.4f}, {acts.max():.4f}]")
        print(f"🔍 Dead Features Debug - Total activations > 0: {(acts > 0).sum().item()}")
    
    # Count how many times each feature is activated
    feature_activations = (acts > 0).sum(dim=0)  # (d_sae,)
    
    if verbose:
        print(f"🔍 Dead Features Debug - Feature activations shape: {feature_activations.shape}")
        print(f"🔍 Dead Features Debug - Feature activations range: [{feature_activations.min():.0f}, {feature_activations.max():.0f}]")
        print(f"🔍 Dead Features Debug - Features with 0 activations: {(feature_activations == 0).sum().item()}")
        print(f"🔍 Dead Features Debug - Features with >0 activations: {(feature_activations > 0).sum().item()}")
    
    # A feature is "dead" if it's never activated
    dead_features = (feature_activations == 0).sum()
    total_features = acts.shape[1]
    
    dead_percentage = (dead_features / total_features) * 100
    
    if verbose:
        print(f"🔍 Dead Features Debug - Dead features: {dead_features.item()}")
        print(f"🔍 Dead Features Debug - Total features: {total_features}")
        print(f"🔍 Dead Features Debug - Dead percentage: {dead_percentage:.4f}%")
    
    return dead_percentage.item()

def calculate_feature_absorption_saebench(weights: Dict) -> float:
    """
    Calculate Feature Absorption following SAEBench methodology
    Uses cosine similarity between decoder weights
    """
    W_dec = weights["W_dec"]  # (d_sae, d_in)
    
    # For large SAEs, sample a subset of features to avoid memory issues
    d_sae = W_dec.shape[0]
    if d_sae > 10000:  # If more than 10k features, sample 10k
        sample_size = 10000
        indices = torch.randperm(d_sae, device=W_dec.device)[:sample_size]
        W_dec = W_dec[indices]
        print(f"🔍 Feature Absorption Debug - Sampling {sample_size} features from {d_sae} total features")
    
    # Convert to numpy for sklearn cosine_similarity
    decoder_weights = W_dec.detach().cpu().numpy()  # (d_sae, d_in)
    
    # Calculate cosine similarities between all pairs of features
    similarities = cosine_similarity(decoder_weights)
    
    # Remove diagonal (self-similarity = 1.0)
    np.fill_diagonal(similarities, 0)
    
    # Calculate absorption as mean of top similarities
    # Following SAEBench: take top k similarities where k = min(100, similarities.shape[0])
    k = min(100, similarities.shape[0])
    top_similarities = np.partition(similarities.flatten(), -k)[-k:]
    absorption = np.mean(top_similarities)
    
    return float(absorption)

def evaluate_sae_health_real_activations(sae_path: str, model_name: str, layer: int,
                                       dataset_name: str, dataset_column: str,
                                       num_samples: int, context_length: int,
                                       batch_size: int, device: torch.device,
                                       verbose: bool = False) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Evaluate SAE health using real model activations"""
    
    print(f"🔍 Evaluating SAE with real model activations")
    print(f"📋 Using dataset: {dataset_name}")
    print(f"🤖 Model: {model_name}, Layer: {layer}")
    
    # Load SAE
    config, weights = load_sae_checkpoint(sae_path, device)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Load dataset samples
    samples = load_dataset_samples(dataset_name, dataset_column, num_samples, 
                                 1000, tokenizer, verbose)
    
    # Tokenize samples
    tokens = tokenize_samples(samples, tokenizer, context_length, device, verbose)
    
    # Extract model activations
    activations = extract_model_activations(model, tokens, layer, device, batch_size, verbose)
    
    # Clean up model to free memory
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Calculate metrics using SAEBench methodology
    print("📈 Calculating Metrics (SAEBench methodology with real activations)...")
    
    # 1. Loss Recovered / FVU
    loss_recovered = calculate_fvu_saebench(activations, weights, verbose)
    
    # 2. L0 Sparsity
    l0_sparsity = calculate_l0_sparsity_saebench(activations, weights)
    
    # 3. Dead Features %
    dead_features = calculate_dead_features_saebench(activations, weights, verbose)
    
    # 4. Feature Absorption
    absorption = calculate_feature_absorption_saebench(weights)
    
    # Compile results
    results = {
        "loss_recovered": loss_recovered,
        "l0_sparsity": l0_sparsity,
        "dead_features_percent": dead_features,
        "feature_absorption": absorption
    }
    
    # Health assessment based on SAEBench findings
    health_assessment = {
        "loss_recovered": {
            "value": loss_recovered,
            "healthy_range": "≥60-70% (SAEBench standard)",
            "is_healthy": loss_recovered >= 60,
            "status": "✅ Healthy" if loss_recovered >= 60 else "❌ Below threshold"
        },
        "l0_sparsity": {
            "value": l0_sparsity,
            "healthy_range": "20 ≤ L0 ≤ 200 (SAEBench sweet spot: 40-120)",
            "is_healthy": 20 <= l0_sparsity <= 200,
            "status": "✅ Healthy" if 20 <= l0_sparsity <= 200 else "❌ Outside range"
        },
        "dead_features": {
            "value": dead_features,
            "healthy_range": "≤10-20% (SAEBench standard)",
            "is_healthy": dead_features <= 20,
            "status": "✅ Healthy" if dead_features <= 20 else "❌ Too many dead features"
        },
        "feature_absorption": {
            "value": absorption,
            "healthy_range": "≤0.25 (SAEBench: 0.25-0.35 = borderline)",
            "is_healthy": absorption <= 0.35,
            "status": "✅ Healthy" if absorption <= 0.25 else "⚠️ Borderline" if absorption <= 0.35 else "❌ High absorption"
        }
    }
    
    return results, health_assessment

def save_results(results: Dict[str, float], health_assessment: Dict[str, Any], 
                output_file: str, args: argparse.Namespace):
    """Save results to JSON file"""
    if output_file is None:
        return
    
    output_data = {
        "results": results,
        "health_assessment": health_assessment,
        "evaluation_config": {
            "sae_path": args.sae_path,
            "model_name": args.model_name,
            "layer": args.layer,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "seed": args.seed
        },
        "methodology": "SAEBench with real model activations"
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Check if SAE path exists
    if not os.path.exists(args.sae_path):
        print(f"❌ SAE path not found: {args.sae_path}")
        return
    
    try:
        # Run evaluation with real activations
        results, health_assessment = evaluate_sae_health_real_activations(
            sae_path=args.sae_path,
            model_name=args.model_name,
            layer=args.layer,
            dataset_name=args.dataset,
            dataset_column=args.dataset_column,
            num_samples=args.num_samples,
            context_length=args.context_length,
            batch_size=args.batch_size,
            device=device,
            verbose=args.verbose
        )
        
        # Print results
        print("\n" + "="*70)
        print("📊 SAE HEALTH EVALUATION RESULTS (Real Model Activations)")
        print("="*70)
        
        for metric, assessment in health_assessment.items():
            print(f"\n🔍 {metric.replace('_', ' ').title()}:")
            if metric in ['loss_recovered', 'dead_features']:
                print(f"   Value: {assessment['value']:.3f}%")
            else:
                print(f"   Value: {assessment['value']:.3f}")
            print(f"   Healthy Range: {assessment['healthy_range']}")
            print(f"   Status: {assessment['status']}")
        
        # Overall health summary
        healthy_metrics = sum(1 for assessment in health_assessment.values() if assessment['is_healthy'])
        total_metrics = len(health_assessment)
        
        print(f"\n" + "="*70)
        print(f"🎯 OVERALL HEALTH: {healthy_metrics}/{total_metrics} metrics healthy")
        
        if healthy_metrics == total_metrics:
            print("✅ SAE is in healthy range across all metrics!")
        elif healthy_metrics >= total_metrics * 0.75:
            print("⚠️ SAE is mostly healthy with some areas for improvement")
        else:
            print("❌ SAE needs improvement in multiple areas")
        
        print("="*70)
        print("\n📚 Methodology: SAEBench with real model activations")
        print(f"🔗 Dataset: {args.dataset}")
        print(f"🤖 Model: {args.model_name} (Layer {args.layer})")
        
        # Save results
        save_results(results, health_assessment, args.output_file, args)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
