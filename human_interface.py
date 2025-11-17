import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import psutil, os, torch

start_time = time.time()


def print_peak_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())

    try:
        cpu_mem_gb = process.memory_info().rss / 1e9  
    except:
        cpu_mem_gb = 0
    
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_mem_gb = cpu_mem_gb + gpu_mem_gb
    
    print(f"[Peak Memory] {step_name}: CPU≈{cpu_mem_gb:.2f} GB | GPU={gpu_mem_gb:.2f} GB | Total≈{total_mem_gb:.2f} GB")

# reset GPU peak memory tracker after measuring
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


# Load model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
base_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
base_model.to(device).eval()

# Load dataset
print("Loading dataset...")
df = pd.read_csv("cleaned_testset_auto.csv")  # expects 'source', 'target', 'label'
df = df[350:]

def split_field(field_name):
    """Split 'endpoint.column' into endpoint (table) and column."""
    parts = field_name.split('.')
    if len(parts) == 2:
        return parts[0], parts[1]

df[['src_endpoint', 'src_column']] = df['source'].apply(lambda x: pd.Series(split_field(x)))
df[['tgt_endpoint', 'tgt_column']] = df['target'].apply(lambda x: pd.Series(split_field(x)))

#Create embeddings
total_tokens = 0

def get_embeddings(texts, batch_size=16):
    global total_tokens
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Count tokens using the tokenizer itself
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        batch_tokens = sum([len(ids) for ids in enc['input_ids']])
        total_tokens += batch_tokens

        # Generate embeddings
        with torch.no_grad():
            outputs = base_model(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)

# Source embeddings
src_endpoint_embeddings = get_embeddings(df['src_endpoint'].astype(str).tolist())
src_column_embeddings   = get_embeddings(df['src_column'].astype(str).tolist())

# Target embeddings
tgt_endpoint_embeddings = get_embeddings(df['tgt_endpoint'].astype(str).tolist())
tgt_column_embeddings   = get_embeddings(df['tgt_column'].astype(str).tolist())
print("created all embeddings")

# Creating cosine similarity scores
endpoint_sim = cosine_similarity(src_endpoint_embeddings, tgt_endpoint_embeddings)
column_sim = cosine_similarity(src_column_embeddings, tgt_column_embeddings)

# ____________________________________________
top_k = 3
endpoint_threshold = 0.7
column_threshold = 0.7
filtered_candidates = []

for i, src in enumerate(df['source']):
    # Step 1: filter by endpoint similarity
    valid_idx = np.where(endpoint_sim[i] >= endpoint_threshold)[0]
    if len(valid_idx) == 0:
        filtered_candidates.append({"source": src, "top_targets": []})
        continue

    # Step 2: filter column similarities only for valid targets
    col_sims = column_sim[i, valid_idx]
    valid_col_idx = np.where(col_sims >= column_threshold)[0]
    if len(valid_col_idx) == 0:
        filtered_candidates.append({"source": src, "top_targets": []})
        continue

    # Step 3: sort top-k by column similarity
    top_idx = col_sims[valid_col_idx].argsort()[::-1][:top_k]
    top_list = [(df['target'].iloc[valid_idx[valid_col_idx[j]]], col_sims[valid_col_idx[j]])
                for j in top_idx]

    filtered_candidates.append({"source": src, "top_targets": top_list})

mapping = {}  # final human-verified mapping

print("\nCandidate Schema Mappings (Human-in-the-loop):\n")

for item in filtered_candidates:
    src = item["source"]
    targets = [t for t, s in item["top_targets"]]  # top-k candidate targets

    # Add "None of the above" as the last option
    options = targets + ["None of the above"]

    if not options:
        print(f"No candidates for source: {src}")
        mapping[src] = "None of the above"
        continue

    # Display options
    print(f"\nSource: {src}")
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")

    # Get user choice
    while True:
        try:
            choice = int(input(f"Select the correct target (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                mapping[src] = options[choice - 1]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Invalid input, enter a number.")

print(mapping)

TP = FP = TN = FN = 0

for _, row in df.iterrows():
    src = row['source']
    tgt = row['target']
    label = row['label']

    pred = mapping.get(src, "None of the above")

    if pred == "None of the above":
        if label == 1:
            FN += 1
        else:
            TN += 1
    else:
        if label == 1:
            if pred == tgt:
                TP += 1
            else:
                FN += 1
        else:  
            FP += 1

print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")


end_time = time.time()
total_time = end_time - start_time
print(f"Total elapsed time: {total_time:.2f} seconds")
print_peak_memory_usage("End of script")
print(f"Total tokens used for embeddings: {total_tokens}")