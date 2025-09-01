#!/bin/bash

echo "🚀 Running Delphi Auto-Interp on Llama-2-7B SAE Models"
echo "========================================================"

# Check if Delphi is installed
if ! python -c "import delphi" 2>/dev/null; then
    echo "📦 Installing Delphi dependencies..."
    
    # Install required packages
    pip install "eai-sparsify>=1.1.3" datasets faiss-cpu sentence-transformers vllm orjson
    
    # Clone and install Delphi
    if [ ! -d "delphi" ]; then
        echo "📥 Cloning Delphi repository..."
        git clone https://github.com/EleutherAI/sae-auto-interp delphi
    fi
    
    echo "🔧 Installing Delphi..."
    cd delphi && pip install -e . && cd ..
fi

echo "✅ Delphi is ready!"

# Define paths
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"

# Check if SAE exists
if [ ! -d "$SAE_PATH" ]; then
    echo "❌ SAE not found at: $SAE_PATH"
    echo "Please check the path and ensure the SAE model is available."
    exit 1
fi

echo "🎯 Base Model: $BASE_MODEL"
echo "🎯 SAE Path: $SAE_PATH"
echo "🎯 Layer: 16"

echo ""
echo "🚀 Running Delphi Auto-Interp..."
echo "Command: python -m delphi $BASE_MODEL $SAE_PATH --n_tokens 50000 --max_latents 20 --hookpoints layers.16 --scorers detection --filter_bos --name llama2_7b_layer16_sae_autointerp_basic"

# Run Delphi with basic WikiText dataset for ultra-fast testing
python -m delphi \
    "$BASE_MODEL" \
    "$SAE_PATH" \
    --n_tokens 50000 \
    --max_latents 20 \
    --hookpoints layers.16 \
    --scorers detection \
    --filter_bos \
    --name "llama2_7b_layer16_sae_autointerp_basic"

echo ""
echo "🎉 Delphi Auto-Interp completed!"
echo "📁 Check the results in: llama2_7b_layer16_sae_autointerp_ultrafast/"
What is the data set on which it has been evaluated it looks good