import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Set up model for speed and precision
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def process_large_file(input_file, chunk_size):
    """Reads a large parquet file in chunks, preserving the original order via indexing."""
    for chunk_idx, chunk in enumerate(
        pd.read_parquet(input_file, chunksize=chunk_size)
    ):
        # Add original index to track order
        chunk["original_index"] = chunk.index
        # Sort by text length
        chunk["text_length"] = chunk["text"].str.len()
        chunk = chunk.sort_values("text_length", ascending=False)
        yield chunk.to_dict("records")


def process_chunk(chunk, batch_size, tokenizer, model, id2label):
    """Process each chunk, predict labels, and save results in original order."""
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]

        texts = [item["text"] for item in batch]
        original_indices = [item["original_index"] for item in batch]

        # Tokenize with dynamic padding to the longest sequence in the batch
        encodings = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move the tokenized batch to the device
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}

        # Run the model on the batch and get logits
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**encodings)
            logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.sigmoid(logits).cpu().tolist()

        # Store each result with its original index
        for idx, prob in zip(original_indices, probs):
            # Create a dictionary of register names and their probabilities
            register_probs = {id2label[str(i)]: round(p, 4) for i, p in enumerate(prob)}
            results.append(
                {
                    "original_index": idx,
                    "register_probabilities": register_probs,
                }
            )

    # Sort results by original index to ensure output order matches input order
    results.sort(key=lambda x: x["original_index"])
    return [result["register_probabilities"] for result in results]


def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True,
        backend="inductor",
    )
    model.eval()

    id2label = model.config.id2label

    total_items = 0
    total_time = 0.0

    # Initialize an empty list to store all results
    all_results = []

    for chunk_idx, chunk in enumerate(
        process_large_file(args.input_file, args.chunk_size)
    ):
        start_time = time.perf_counter()

        results = process_chunk(
            chunk,
            args.batch_size,
            tokenizer,
            model,
            id2label,
        )

        # Add results to the list
        all_results.extend(results)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        throughput = len(chunk) / elapsed_time if elapsed_time > 0 else float("inf")

        # Update totals
        total_items += len(chunk)
        total_time += elapsed_time
        average_throughput = (
            total_items / total_time if total_time > 0 else float("inf")
        )

        # Log progress every 100 chunks
        if chunk_idx % 100 == 0:
            print(
                f"Chunk {chunk_idx}: Throughput = {throughput:.2f} items/s, "
                f"Average Throughput = {average_throughput:.2f} items/s"
            )

    # Convert results to DataFrame
    df_results = pd.DataFrame(all_results)

    # Convert register_probabilities from dict to separate columns
    final_df = pd.json_normalize(df_results)

    # Save just the register probabilities to parquet
    final_df.to_parquet(args.output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input parquet file.")
    parser.add_argument(
        "output_file", type=str, help="Path to the output parquet file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TurkuNLP/web-register-classification-en",
        help="Model to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of items per batch.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Number of items per chunk.",
    )
    args = parser.parse_args()
    main(args)
