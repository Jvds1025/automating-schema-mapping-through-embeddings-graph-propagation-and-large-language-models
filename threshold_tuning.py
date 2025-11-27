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
    # CPU peak memory is platform dependent; on Linux, can use rss peak from /proc
    try:
        cpu_mem_gb = process.memory_info().rss / 1e9  # instantaneous CPU memory
    except:
        cpu_mem_gb = 0
    
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_mem_gb = cpu_mem_gb + gpu_mem_gb
    
    print(f"[Peak Memory] {step_name}: CPU≈{cpu_mem_gb:.2f} GB | GPU={gpu_mem_gb:.2f} GB | Total≈{total_mem_gb:.2f} GB")

# Optional: reset GPU peak memory tracker after measuring
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
df = df[50:100]

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

alpha = 0.3
beta = 0.7
combined_sim = ((alpha *endpoint_sim) + (beta *column_sim))  # simple average

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_predictions(pred_dict, df):
    TP1 = FP1 = FN1 = TN1 = 0
    TP3 = FP3 = FN3 = TN3 = 0

    for _, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        label = row["label"]
        preds = pred_dict.get(src, [])

        top1 = preds[:1]
        top3 = preds[:3]

        #  TOP 1 
        if label == 1:
            if tgt in top1:
                TP1 += 1
            else:
                FN1 += 1
        else:
            if len(top1) > 0:
                FP1 += 1
            else:
                TN1 += 1

        # TOP 3 
        if label == 1:
            if tgt in top3:
                TP3 += 1
            else:
                FN3 += 1
        else:
            if len(top3) > 0:
                FP3 += 1
            else:
                TN3 += 1

    def compute(TP, FP, FN, TN):
        acc = (TP + TN) / (TP + FP + FN + TN)
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return acc, prec, rec, f1

    return compute(TP1, FP1, FN1, TN1), compute(TP3, FP3, FN3, TN3)
    
# THRESHOLD TUNING
endpoint_grid = np.arange(0.1, 0.95, 0.025)
column_grid   = np.arange(0.1, 0.95, 0.025)

results = []

for ep_t in endpoint_grid:
    for col_t in column_grid:
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

            # Step 3: rank
            top_idx = col_sims[valid_col_idx].argsort()[::-1][:5]
            top_list = [(df["target"].iloc[valid_idx[valid_col_idx[j]]])
                        for j in top_idx]

            filtered.append({"source": src, "top_targets": top_list})

        # Build prediction dictionary
        pred_dict = {item["source"]: item["top_targets"] for item in filtered}

        # Evaluate
        top1_metrics, top3_metrics = evaluate_predictions(pred_dict, df)

        results.append({
            "endpoint_threshold": ep_t,
            "column_threshold": col_t,
            "top1_accuracy": top1_metrics[0],
            "top1_precision": top1_metrics[1],
            "top1_recall": top1_metrics[2],
            "top1_f1": top1_metrics[3],
            "top3_accuracy": top3_metrics[0],
            "top3_precision": top3_metrics[1],
            "top3_recall": top3_metrics[2],
            "top3_f1": top3_metrics[3],
        })

# Convert to DataFrame
tuning_df = pd.DataFrame(results)

print("\n TOP 1 BEST F1")
best_top1 = tuning_df.loc[tuning_df["top1_f1"].idxmax()]
print(best_top1)

print("\n TOP 3 BEST F1 ")
best_top3 = tuning_df.loc[tuning_df["top3_f1"].idxmax()]
print(best_top3)

# Save results
tuning_df.to_csv("threshold_tuning_results.csv", index=False)
print("\nSaved full grid search to threshold_tuning_results.csv")
