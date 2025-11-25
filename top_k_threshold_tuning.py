import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import psutil
import os
import sys

# Define the search space for all three hyperparameters
ENDPOINT_GRID = np.arange(0.5, 0.9, 0.05) 
COLUMN_GRID   = np.arange(0.6, 0.9, 0.05)  
TOP_K_GRID    = [1, 2, 3, 4, 5, 10, 20]   

start_time = time.time()


def print_peak_memory_usage(step_name=""):
    """Prints current CPU memory and peak GPU memory usage."""
    process = psutil.Process(os.getpid())
    try:
        cpu_mem_gb = process.memory_info().rss / 1e9  
    except:
        cpu_mem_gb = 0
    
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_mem_gb = cpu_mem_gb + gpu_mem_gb
    
    print(f"[Peak Memory] {step_name}: CPU≈{cpu_mem_gb:.2f} GB | GPU={gpu_mem_gb:.2f} GB | Total≈{total_mem_gb:.2f} GB")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

def split_field(field_name):
    """Split 'endpoint.column' into endpoint (table) and column."""
    parts = field_name.split('.')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

# Load model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
    base_model = AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-4B",
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    base_model.to(device).eval()
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)


# Dataset Loading and Preprocessing

print("Loading dataset...")
try:
    df = # Load your dasaset
    
    df = df[0:50].reset_index(drop=True) # Slice dataset if necessary
except FileNotFoundError:
    print("Error: dataset not found.", file=sys.stderr)
    sys.exit(1)


# Split source/target into endpoint and column parts
df[['src_endpoint', 'src_column']] = df['source'].apply(lambda x: pd.Series(split_field(x)))
df[['tgt_endpoint', 'tgt_column']] = df['target'].apply(lambda x: pd.Series(split_field(x)))


# Embedding Function (copied for completeness)

total_tokens = 0
def get_embeddings(texts, batch_size=16):
    """Generates embeddings for a list of texts in batches."""
    global total_tokens
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        batch_tokens = sum([len(ids) for ids in enc['input_ids']])
        total_tokens += batch_tokens
        with torch.no_grad():
            outputs = base_model(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)

# Embedding Generation & Similarity Calculation

print("Creating embeddings...")
src_endpoint_embeddings = get_embeddings(df['src_endpoint'].astype(str).tolist())
src_column_embeddings   = get_embeddings(df['src_column'].astype(str).tolist())
tgt_endpoint_embeddings = get_embeddings(df['tgt_endpoint'].astype(str).tolist())
tgt_column_embeddings   = get_embeddings(df['tgt_column'].astype(str).tolist())
print("Created all embeddings.")

# Creating cosine similarity scores
endpoint_sim = cosine_similarity(src_endpoint_embeddings, tgt_endpoint_embeddings)
column_sim = cosine_similarity(src_column_embeddings, tgt_column_embeddings)


# Evaluation Function (Modified for flexible K) 

def evaluate_predictions(pred_dict, df, k_value):
    """Calculates TP/FP/FN/TN and derived metrics for a specific K."""
    TPk = FPk = FNk = TNk = 0

    for _, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        label = row["label"]
        preds = pred_dict.get(src, [])
        topk = preds[:k_value]

        match_topk = tgt in topk

        if label == 1:
            if match_topk:
                TPk += 1
            else:
                FNk += 1
        else:  
            if len(topk) > 0:
                FPk += 1
            else:
                TNk += 1

    def compute(TP, FP, FN, TN):
        """Helper to compute metrics from counts."""
        total = TP + FP + FN + TN
        acc = (TP + TN) / total if total > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return acc, prec, rec, f1, TP, FP, FN, TN

    return compute(TPk, FPk, FNk, TNk)


# COMBINED GRID SEARCH / TUNING 

print("\nStarting Combined Threshold and Top-K Grid Search...")
print(f"Testing {len(ENDPOINT_GRID) * len(COLUMN_GRID) * len(TOP_K_GRID)} total combinations...")

results = []

# Outer loops for thresholds
for ep_t in ENDPOINT_GRID:
    for col_t in COLUMN_GRID:
        
        # Inner loop for Top-K
        for k in TOP_K_GRID:

            filtered = []

            for i, src in enumerate(df["source"]):
                # Step 1: endpoint filter
                valid_idx = np.where(endpoint_sim[i] >= ep_t)[0]
                if len(valid_idx) == 0:
                    filtered.append({"source": src, "top_targets": []})
                    continue

                # Step 2: column filter
                col_sims = column_sim[i, valid_idx]
                valid_col_idx = np.where(col_sims >= col_t)[0]
                if len(valid_col_idx) == 0:
                    filtered.append({"source": src, "top_targets": []})
                    continue

                # Step 3: rank and take TOP-K (using the current 'k' value)
                top_idx = col_sims[valid_col_idx].argsort()[::-1][:k]
                
                # Map the indices back to the original target names
                top_list = [(df["target"].iloc[valid_idx[valid_col_idx[j]]])
                            for j in top_idx]

                filtered.append({"source": src, "top_targets": top_list})

            # Build prediction dictionary
            pred_dict = {item["source"]: item["top_targets"] for item in filtered}

            # Evaluate this set of parameters 
            acc, prec, rec, f1, TP, FP, FN, TN = evaluate_predictions(pred_dict, df, k)

            results.append({
                "K_value": k,
                "endpoint_threshold": ep_t,
                "column_threshold": col_t,
                "TopK_F1": f1,
                "TopK_Recall": rec,
                "TopK_Precision": prec,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
            })

# Final Results and Output 

# Convert results to DataFrame
tuning_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print(f"OPTIMAL PARAMETER COMBINATION (Grid Search)")

# Find the row with the maximum F1 score across ALL combinations
best_combo = tuning_df.loc[tuning_df["TopK_F1"].idxmax()]

print(f"Optimal K Value (by Max F1): K={best_combo['K_value']:.0f}")
print(f"Optimal EP Threshold: {best_combo['endpoint_threshold']:.3f}")
print(f"Optimal COL Threshold: {best_combo['column_threshold']:.3f}")
print("-" * 30)
print(f"Resulting F1-score: {best_combo['TopK_F1']:.4f}")
print(f"Resulting Recall: {best_combo['TopK_Recall']:.4f}")
print(f"Resulting Precision: {best_combo['TopK_Precision']:.4f}")
print("=" * 80)

# Save results
tuning_df.to_csv("combined_tuning_results.csv", index=False)
print("\nSaved full grid search results to combined_tuning_results.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Total elapsed time: {total_time:.2f} seconds")
print_peak_memory_usage("End of script")

print(f"Total tokens used for embeddings: {total_tokens}")
