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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lora_model_path = "best_model.pt"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
base_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load LoRA fine-tuned weights
try:
    state_dict = torch.load(lora_model_path, map_location=device)
    base_model.load_state_dict(state_dict, strict=False)
    model = base_model
    print("Loaded LoRA adapter weights.")
except Exception:
    print("Could not load LoRA weights, using base model.")
    model = base_model

model.to(device)
model.eval()

# Load dataset
print("Loading dataset...")
df = pd.read_csv("cleaned_testset_auto.csv")  # expects 'source', 'target', 'label'
df = df[0:50]

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

#Creating cosine similarity scores
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

# Print results
for item in filtered_candidates:
    print(f"Source: {item['source']}")
    if not item['top_targets']:
        print("  No candidates passed endpoint or column threshold.")
    else:
        for rank, (target, score) in enumerate(item['top_targets'], start=1):
            print(f"  Top {rank}: {target} | Similarity: {score:.3f}")
    print("-" * 40)

# Create prediction dictionary
pred_dict = {item['source']: [t for t, s in item['top_targets']] for item in filtered_candidates}
print("Dictionary created.")


# Initialize counters
TP_Top1 = FP_Top1 = TN_Top1 = FN_Top1 = 0
TP_Top3 = FP_Top3 = TN_Top3 = FN_Top3 = 0

# Evaluate predictions
for _, row in df.iterrows():
    src = row['source']
    tgt = row['target']
    label = row['label']

    preds = pred_dict.get(src, [])
    top1_preds = preds[:1]  # top-1
    top3_preds = preds[:3]  # top-3

    # TOP 1 evaluation 
    if label == 1:
        if any(p == tgt or p == src for p in top1_preds):
            TP_Top1 += 1
        else:
            FN_Top1 += 1
    else:  # label == 0
        if top1_preds:
            FP_Top1 += 1
        else:
            TN_Top1 += 1

    # TOP 3 evaluation 
    if label == 1:
        if any(p == tgt or p == src for p in top3_preds):
            TP_Top3 += 1
        else:
            FN_Top3 += 1
    else:
        if top3_preds:
            FP_Top3 += 1
        else:
            TN_Top3 += 1

# Print evaluation results 
print(f"Top-1 → TP: {TP_Top1}, FP: {FP_Top1}, FN: {FN_Top1}, TN: {TN_Top1}")
print(f"Top-3 → TP: {TP_Top3}, FP: {FP_Top3}, FN: {FN_Top3}, TN: {TN_Top3}")

print(pred_dict)

end_time = time.time()
total_time = end_time - start_time
print(f"Total elapsed time: {total_time:.2f} seconds")
print_peak_memory_usage("End of script")
print(f"Total tokens used for embeddings: {total_tokens}")

