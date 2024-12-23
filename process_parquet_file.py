import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import pyarrow.parquet as pq
from fastparquet import ParquetFile, write

# Set up model for speed and precision
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def process_large_file(input_file, chunk_size):
    """Reads a large parquet file in chunks efficiently."""

    current_position = 0
    pf = pq.ParquetFile(input_file)

    # Read in our specified chunk sizes
    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        # Convert chunk to list of dicts and add indices
        chunk = []
        for i, row in df.iterrows():
            document = row.to_dict()
            document["original_index"] = current_position + i
            chunk.append(document)

        # Sort by text length
        chunk.sort(key=lambda x: len(x["text"]), reverse=True)

        yield chunk
        current_position += len(df)

        # Clear DataFrame to free memory
        del df


def process_chunk(chunk, batch_size, tokenizer, model, id2label):
    """Process each chunk and return results immediately."""
    results = []
    for i in range(0, len(chunk), batch_size):
        batch = chunk[i : i + batch_size]
        texts = [item["text"] for item in batch]
        original_indices = [item["original_index"] for item in batch]
        ids = [item["id"] for item in batch]  # Get the original 'id'

        encodings = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encodings = {key: tensor.to(device) for key, tensor in encodings.items()}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**encodings)
            logits = outputs.logits

        probs = torch.sigmoid(logits).cpu().tolist()

        for idx, id_, prob in zip(original_indices, ids, probs):  # Include id_ in loop
            register_probs = {id2label[i]: round(p, 4) for i, p in enumerate(prob)}
            results.append(
                {
                    "original_index": idx,
                    "id": id_,  # Include id in the result
                    "register_probabilities": register_probs,
                }
            )

    results.sort(key=lambda x: x["original_index"])
    return [
        {"id": result["id"], "register_probabilities": result["register_probabilities"]}
        for result in results
    ]


def write_incremental_parquet(results, output_file, id2label, first_write=False):
    """Write results incrementally to parquet file."""

    # Create a list of dictionaries with the desired structure
    formatted_results = []
    for result in results:
        row = {"id": result["id"]}
        # Add probabilities in the order specified by id2label
        probs = result["register_probabilities"]
        for idx in range(len(id2label)):
            label = id2label[idx]
            row[label] = probs[label]
        formatted_results.append(row)

    # Convert to DataFrame - no need for json_normalize now
    df = pd.DataFrame(formatted_results)

    if first_write:
        df.to_parquet(output_file, index=False, engine="fastparquet")
    else:
        pf = ParquetFile(output_file)
        write(output_file, df, append=True, file_scheme="simple")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
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
    first_write = True

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

        # Write results immediately instead of storing them
        write_incremental_parquet(results, args.output_file, id2label, first_write)
        first_write = False

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        throughput = len(chunk) / elapsed_time if elapsed_time > 0 else float("inf")

        total_items += len(chunk)
        total_time += elapsed_time
        average_throughput = (
            total_items / total_time if total_time > 0 else float("inf")
        )

        if chunk_idx % 100 == 0:
            print(
                f"Chunk {chunk_idx}: Throughput = {throughput:.2f} items/s, "
                f"Average Throughput = {average_throughput:.2f} items/s"
            )


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
        "--base_model",
        type=str,
        default="xlm-roberta-large",
        help="Base model",
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
