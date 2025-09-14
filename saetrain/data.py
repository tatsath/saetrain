"""Tools for tokenizing and manipulating text datasets."""

import math
from multiprocessing import cpu_count
from typing import TypeVar, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    dataset_name: str = None,
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """
    
    # Intelligent text column detection and preprocessing
    if "text" not in data.column_names:
        print("🔍 Dataset doesn't have 'text' column - using intelligent detection...")
        
        # Detect the appropriate text column
        detected_text_key = detect_text_column(data, dataset_name=dataset_name)
        
        # Preprocess the dataset to have a 'text' column
        data = preprocess_dataset_for_text(data, detected_text_key, dataset_name=dataset_name)
        
        # Update text_key to use the standard 'text' column
        text_key = "text"
        print(f"✅ Dataset now has 'text' column ready for tokenization")

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


def detect_text_column(dataset: Dataset, dataset_name: str = None) -> str:
    """
    Intelligently detect the text column in a dataset using SAE Lens-style approach.
    
    Args:
        dataset: The dataset to analyze
        dataset_name: Optional dataset name for specific handling
        
    Returns:
        The detected text column name
    """
    available_columns = dataset.column_names
    print(f"🔍 Analyzing dataset columns: {available_columns}")
    
    # Dataset-specific mappings (like SAEBench)
    dataset_specific_mappings = {
        "HuggingFaceH4/ultrachat_200k": "messages",
        "lmsys/lmsys-chat-1m": "conversation", 
        "LabHC/bias_in_bios": "hard_text",
        "canrager/amazon_reviews_mcauley_1and5": "text",
        "EleutherAI/pile": "text",
        "wikitext": "text",
        "openwebtext": "text",
    }
    
    # Check for dataset-specific mapping first
    if dataset_name and dataset_name in dataset_specific_mappings:
        expected_column = dataset_specific_mappings[dataset_name]
        if expected_column in available_columns:
            print(f"✅ Using dataset-specific column: '{expected_column}' for {dataset_name}")
            return expected_column
    
    # Intelligent text column detection (SAE Lens approach)
    text_keywords = ['text', 'content', 'sentence', 'question', 'context', 'article', 'passage', 'document']
    
    for keyword in text_keywords:
        for col in available_columns:
            if keyword in col.lower():
                print(f"✅ Found text column: '{col}' (matched keyword: '{keyword}')")
                return col
    
    # Check for chat-specific columns
    chat_keywords = ['messages', 'conversation', 'dialogue', 'chat']
    for keyword in chat_keywords:
        for col in available_columns:
            if keyword in col.lower():
                print(f"✅ Found chat column: '{col}' (matched keyword: '{keyword}')")
                return col
    
    # Fallback: use first available column
    if available_columns:
        fallback_column = available_columns[0]
        print(f"⚠️ No obvious text column found. Using first column: '{fallback_column}'")
        return fallback_column
    
    raise ValueError("No columns found in dataset")


def preprocess_dataset_for_text(dataset: Dataset, text_column: str, dataset_name: str = None) -> Dataset:
    """
    Preprocess dataset to extract text content, handling different formats.
    
    Args:
        dataset: The dataset to preprocess
        text_column: The detected text column
        dataset_name: Optional dataset name for specific handling
        
    Returns:
        Preprocessed dataset with 'text' column
    """
    print(f"🔄 Preprocessing dataset using column: '{text_column}'")
    
    # Handle chat datasets with conversation format
    if text_column in ['messages', 'conversation']:
        print("💬 Detected chat dataset format - extracting conversation text...")
        
        def extract_chat_text(example):
            """Extract text from chat conversation format."""
            messages = example[text_column]
            text_parts = []
            
            if isinstance(messages, list):
                # Handle list of message dictionaries
                for message in messages:
                    if isinstance(message, dict):
                        role = message.get('role', 'user')
                        content = message.get('content', '')
                        text_parts.append(f'{role}: {content}')
                    else:
                        text_parts.append(str(message))
            else:
                # Handle other formats
                text_parts.append(str(messages))
            
            # Join with double newlines for clarity
            full_text = '\n\n'.join(text_parts)
            return {'text': full_text}
        
        processed_dataset = dataset.map(extract_chat_text, remove_columns=dataset.column_names)
        print(f"✅ Successfully extracted text from {len(processed_dataset)} conversations")
        return processed_dataset
    
    # Handle regular text datasets
    elif text_column == 'text':
        print("📝 Dataset already has 'text' column - no preprocessing needed")
        return dataset
    
    # Handle other text columns
    else:
        print(f"📝 Converting '{text_column}' column to 'text' column...")
        
        def rename_text_column(example):
            """Rename the text column to 'text'."""
            return {'text': example[text_column]}
        
        processed_dataset = dataset.map(rename_text_column, remove_columns=dataset.column_names)
        print(f"✅ Successfully converted {len(processed_dataset)} examples")
        return processed_dataset


class MemmapDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""

    def __init__(
        self,
        data_path: str,
        ctx_len: int,
        max_examples: int | None = None,
        dtype=np.uint16,
    ):
        mmap = np.memmap(data_path, dtype=dtype, mode="r").reshape(-1, ctx_len)
        self.mmap = mmap[:max_examples]

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return dict(input_ids=torch.from_numpy(self.mmap[idx].astype(np.int64)))

    def select(self, rng: range) -> "MemmapDataset":
        """Select a subset of the dataset."""
        mmap = MemmapDataset.__new__(MemmapDataset)
        mmap.mmap = self.mmap[rng.start : rng.stop]
        return mmap

    def shard(self, num_shards: int, shard_id: int) -> "MemmapDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        mmap = MemmapDataset.__new__(MemmapDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap
