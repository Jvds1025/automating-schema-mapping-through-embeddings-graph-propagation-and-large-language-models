# Schema Auto-Mapping with Graph Propagation (Embed First + Metrics)

import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import json

# Setup
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
df = pd.read_csv("cleaned_testset_auto.csv")  # assumes 'source' & 'target' columns

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
base_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
base_model.to(device).eval()

# ____________________________________________________________________________________________
# Compute embeddings

def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = base_model(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)

print("Computing embeddings for all sources and targets")
all_source_texts = df["source"].tolist()
all_target_texts = df["target"].tolist()

source_embeds_all = get_embeddings(all_source_texts, batch_size=16)
target_embeds_all = get_embeddings(all_target_texts, batch_size=16)
print("Embeddings computed.")

# _____________________________________________________________________
# Step 1: Split into batches
batch_ranges = [(i, i + 50) for i in range(0, 350, 50)]
batch_ranges.append((350, len(df)))  # last batch may be shorter

all_auto_mappings = {}
top_k = 3
threshold = 0.72
alpha = 0.3

all_top1_preds = []
all_true_targets = []
all_top3_hits = 0
total_rows = 0

# _____________________________________________________________________
# Step 2: Process each batch
for idx, (start, end) in enumerate(batch_ranges):
    print(f"\n=== Processing batch {idx+1} (rows {start}:{end}) ===")
    batch_df = df[start:end]
    source_schema = batch_df['source'].tolist()
    target_schema = batch_df['target'].tolist()

    # Extract embeddings for this batch
    source_embeds = source_embeds_all[start:end]
    target_embeds = target_embeds_all[start:end]

    # Compute similarity
    similarity_matrix = cosine_similarity(source_embeds, target_embeds)

    # Graph propagation
    propagated_source_embeddings = []
    for i, s_node in enumerate(source_schema):
        neighbors = [(target_embeds[j], similarity_matrix[i][j])
                     for j in range(len(target_schema)) if similarity_matrix[i][j] > threshold]

        if not neighbors:
            propagated_source_embeddings.append(source_embeds[i])
            continue

        neighbor_embeds = np.array([e for (e, _) in neighbors])
        weights = np.array([w for (_, w) in neighbors])
        weighted_avg = np.average(neighbor_embeds, axis=0, weights=weights)
        new_embed = (1 - alpha) * source_embeds[i] + alpha * weighted_avg
        propagated_source_embeddings.append(new_embed)

    propagated_source_embeddings = np.vstack(propagated_source_embeddings)
    propagated_similarity_matrix = cosine_similarity(propagated_source_embeddings, target_embeds)
    print("✅ Propagation complete.")

    # Top-k mappings
    auto_mapping = {}
    for i, s_node in enumerate(source_schema):
        sim_scores = [(target_schema[j], propagated_similarity_matrix[i][j]) for j in range(len(target_schema))]
        top_matches = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_k]
        best_target, best_score = top_matches[0]
        auto_mapping[s_node] = best_target

        # Metrics
        all_top1_preds.append(best_target)
        all_true_targets.append(target_schema[i])
        if target_schema[i] in [t for t,_ in top_matches]:
            all_top3_hits += 1
        total_rows += 1

        print(f"{s_node} → {best_target} (score: {best_score:.3f})")

    all_auto_mappings[f"batch_{idx+1}"] = auto_mapping

# _____________________________________________________________________________________________________
# Step 3: Compute evaluation metrics
df = """your dataset"""
mapping = {} #Add here the mapping dictionary
df_batch = pd.DataFrame(list(mapping.items()), columns=['source', 'predicted_target_field'])
df_merged = pd.merge(df, df_batch, on='source', how='left')

df_merged["pred_correct"] = (df_merged["predicted_target_field"] == df_merged["target"]).astype(int)
total = len(df_merged)
correct = ((df_merged["pred_correct"]) & (df_merged["label"] == 1) & 
           (df_merged["target"] == df_merged["predicted_target_field"])).sum()
true_positives = correct
false_positives = ((df_merged["pred_correct"] == 1) & (df_merged["label"] == 0)).sum()
false_negatives = ((df_merged["pred_correct"] == 0) & (df_merged["label"] == 1)).sum()
true_negatives = ((df_merged["pred_correct"] == 0) & (df_merged["label"] == 0)).sum()

precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (true_positives + false_negatives + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
accuracy = (true_positives + true_negatives) / total

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# ____________________________________________________________________________
# Step 4: Print all batch mappings
print("\n✅ Final dictionary of propagated mappings for all batches:")
for batch_name, mapping in all_auto_mappings.items():
    print(f"\n{batch_name}:")
    for s, t in mapping.items():
        print(f"  {s} → {t}")

with open("all_auto_mappings.json", "w", encoding="utf-8") as f:
    json.dump(all_auto_mappings, f, indent=4, ensure_ascii=False)

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")


