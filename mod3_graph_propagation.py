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

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Load model & tokenizer
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
df = pd.read_csv("cleaned_testset_auto.csv")  
df = df[350:].reset_index(drop=True)

def split_field(field_name):
    parts = field_name.split('.')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

df[['src_endpoint', 'src_column']] = df['source'].apply(lambda x: pd.Series(split_field(x)))
df[['tgt_endpoint', 'tgt_column']] = df['target'].apply(lambda x: pd.Series(split_field(x)))

for col in ['src_endpoint', 'src_column', 'tgt_endpoint', 'tgt_column']:
    df[col] = df[col].replace("None", "UNKNOWN").fillna("UNKNOWN")


# Get embeddings 
total_tokens = 0

def get_embeddings(texts, batch_size=16):
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


# Create embeddings
src_endpoint_embeddings = get_embeddings(df['src_endpoint'].astype(str).tolist())
src_column_embeddings   = get_embeddings(df['src_column'].astype(str).tolist())
tgt_endpoint_embeddings = get_embeddings(df['tgt_endpoint'].astype(str).tolist())
tgt_column_embeddings   = get_embeddings(df['tgt_column'].astype(str).tolist())
print("created all embeddings")
print_peak_memory_usage("After embeddings")

# Raw similarities
endpoint_sim_raw = cosine_similarity(src_endpoint_embeddings, tgt_endpoint_embeddings)  
column_sim_raw = cosine_similarity(src_column_embeddings, tgt_column_embeddings)


# table-level embeddings & mappings

# We want unique src/tgt endpoints and averaged embeddings per endpoint
unique_src_endpoints, src_inv = np.unique(df['src_endpoint'].values, return_inverse=True)
unique_tgt_endpoints, tgt_inv = np.unique(df['tgt_endpoint'].values, return_inverse=True)

# Average endpoint embeddings per unique endpoint (source)
src_table_embeddings = np.zeros((len(unique_src_endpoints), src_endpoint_embeddings.shape[1]), dtype=np.float32)
for idx, name in enumerate(unique_src_endpoints):
    rows = np.where(df['src_endpoint'].values == name)[0]
    src_table_embeddings[idx] = src_endpoint_embeddings[rows].mean(axis=0)

# Average endpoint embeddings per unique endpoint (target)
tgt_table_embeddings = np.zeros((len(unique_tgt_endpoints), tgt_endpoint_embeddings.shape[1]), dtype=np.float32)
for idx, name in enumerate(unique_tgt_endpoints):
    rows = np.where(df['tgt_endpoint'].values == name)[0]
    tgt_table_embeddings[idx] = tgt_endpoint_embeddings[rows].mean(axis=0)

# Table-level similarity (initial)
table_sim = cosine_similarity(src_table_embeddings, tgt_table_embeddings)  


# TABLE-LEVEL PROPAGATION
def row_normalize(mat, eps=1e-9):
    rs = mat.sum(axis=1, keepdims=True)
    return mat / (rs + eps)

def propagate_table_sim(table_sim, src_table_emb, tgt_table_emb, threshold=0.0, gamma=0.7):
    """
    Propagate on table-level graph:
      - Build adjacency for src tables and tgt tables using cosine on table embeddings
      - Normalize and diffuse table_sim: A_src @ table_sim @ A_tgt.T
      - Blend with original table_sim via gamma (anchor)
    """
    # adjacency among source tables (self-similarities)
    A_src = cosine_similarity(src_table_emb, src_table_emb)
    A_tgt = cosine_similarity(tgt_table_emb, tgt_table_emb)

    # optional thresholding to sparsify
    if threshold > 0:
        A_src[A_src < threshold] = 0
        A_tgt[A_tgt < threshold] = 0

    A_src = row_normalize(A_src)
    A_tgt = row_normalize(A_tgt)

    propagated = A_src @ table_sim @ A_tgt.T
    refined = gamma * table_sim + (1 - gamma) * propagated
    return refined

# run table propagation
table_sim_refined = propagate_table_sim(table_sim, src_table_embeddings, tgt_table_embeddings, threshold=0.0, gamma=0.75)
print("table_sim refined shape:", table_sim_refined.shape)

# Expand refined table sim back to per-row endpoint similarity matrix
src_table_idx_per_row = src_inv  
tgt_table_idx_per_row = tgt_inv  #

n = len(df)
endpoint_sim_refined = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    si = src_table_idx_per_row[i]
    for j in range(n):
        tj = tgt_table_idx_per_row[j]
        endpoint_sim_refined[i, j] = table_sim_refined[si, tj]

# Diagnostic difference
print("Endpoint sim mean change:",
      float(np.abs(endpoint_sim_refined - endpoint_sim_raw).mean()))


# Column-level blockwise propagation
from collections import defaultdict

def propagate_column_sim_blockwise_safe(col_sim, src_endpoints, tgt_endpoints, alpha=0.55):
    from collections import defaultdict
    src_groups = defaultdict(list)
    tgt_groups = defaultdict(list)

    for i, s in enumerate(src_endpoints):
        src_groups[s].append(i)
    for j, t in enumerate(tgt_endpoints):
        tgt_groups[t].append(j)

    refined = col_sim.copy()

    for s_table, s_idx in src_groups.items():
        for t_table, t_idx in tgt_groups.items():
            s_idx = np.array(s_idx, dtype=int)
            t_idx = np.array(t_idx, dtype=int)
            block = col_sim[np.ix_(s_idx, t_idx)]
            if block.size == 0:
                continue

            if block.shape[0] == 1 or block.shape[1] == 1:
                refined[np.ix_(s_idx, t_idx)] = block
                continue

            # Row smoothing: 
            row_mean = block.mean(axis=1, keepdims=True)
            row_smooth = np.tile(row_mean, (1, block.shape[1]))

            # Column smoothing:
            col_mean = block.mean(axis=0, keepdims=True)
            col_smooth = np.tile(col_mean, (block.shape[0], 1))

            # Combine
            smoothed = (row_smooth + col_smooth) / 2.0

            refined[np.ix_(s_idx, t_idx)] = alpha * block + (1 - alpha) * smoothed

    return refined




# Run blockwise column propagation
column_sim_refined = propagate_column_sim_blockwise_safe(column_sim_raw, df['src_endpoint'].tolist(), df['tgt_endpoint'].tolist(), alpha=0.55)
print("column_sim refined shape:", column_sim_refined.shape)
print("Column sim mean change:", float(np.abs(column_sim_refined - column_sim_raw).mean()))

print_peak_memory_usage("After propagation refinements")


# Combine refined endpoint and column sims into final combined_sim

# weights for combination (tune on dev set)
w_endpoint = 0.3
w_column = 0.7

combined_sim_refined = w_endpoint * endpoint_sim_refined + w_column * column_sim_refined

# 
# Compare to original combined (original used alpha/beta earlier)
original_combined_sim = 0.3 * endpoint_sim_raw + 0.7 * column_sim_raw
print("Combined sim mean change:", float(np.abs(combined_sim_refined - original_combined_sim).mean()))

# Now apply filtering & top-k selection (unchanged logic, but using refined combined sim)
top_k = 3
endpoint_threshold = 0.7
column_threshold = 0.7
filtered_candidates = []

for i, src in enumerate(df['source']):
    # Step 1: filter by endpoint similarity (use refined endpoint_sim_refined)
    valid_idx = np.where(endpoint_sim_refined[i] >= endpoint_threshold)[0]
    if len(valid_idx) == 0:
        filtered_candidates.append({"source": src, "top_targets": []})
        continue

    # Step 2: filter column similarities only for valid targets (use refined column sim)
    col_sims = column_sim_refined[i, valid_idx]
    valid_col_idx = np.where(col_sims >= column_threshold)[0]
    if len(valid_col_idx) == 0:
        filtered_candidates.append({"source": src, "top_targets": []})
        continue

    # Step 3: sort top-k by column similarity (you may choose to rank by combined_sim_refined as well)
    top_idx = col_sims[valid_col_idx].argsort()[::-1][:top_k]
    top_list = [(df['target'].iloc[valid_idx[valid_col_idx[j]]], float(col_sims[valid_col_idx[j]]))
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
print("Candidate dictionary created.")


# Evaluation (same as before)
TP_Top1 = FP_Top1 = TN_Top1 = FN_Top1 = 0
TP_Top3 = FP_Top3 = TN_Top3 = FN_Top3 = 0

for _, row in df.iterrows():
    src = row['source']
    tgt = row['target']
    label = row['label']

    preds = pred_dict.get(src, [])
    top1_preds = preds[:1]  
    top3_preds = preds[:3]  

    # TOP 1 evaluation 
    if label == 1:
        if any(p == tgt or p == src for p in top1_preds):
            TP_Top1 += 1
        else:
            FN_Top1 += 1
    else:  
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

print(f"Refined Top-1 → TP: {TP_Top1}, FP: {FP_Top1}, FN: {FN_Top1}, TN: {TN_Top1}")
print(f"Refined Top-3 → TP: {TP_Top3}, FP: {FP_Top3}, FN: {FN_Top3}, TN: {TN_Top3}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total elapsed time: {total_time:.2f} seconds")
print_peak_memory_usage("End of script")
print(f"Total tokens used for embeddings: {total_tokens}")


