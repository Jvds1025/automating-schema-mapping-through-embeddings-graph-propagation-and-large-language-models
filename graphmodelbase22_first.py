# ===============================================================
# Schema Auto-Mapping with Graph Propagation (Fixed Batches + Metrics)
# ===============================================================
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import psutil, os

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
df = pd.read_csv("cleaned_testset_auto.csv")  # assumes 'source' & 'target' columns exist

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
base_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
base_model.to(device).eval()

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = base_model(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)



# ---------------------------------------------------------------
# Fixed batch ranges
# ---------------------------------------------------------------
batch_ranges = [(i, i + 50) for i in range(0, 350, 50)]
batch_ranges.append((350, len(df)))  # last batch may be shorter

all_auto_mappings = {}
all_top1_preds = []
all_true_targets = []

# ---------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------
for idx, (start, end) in enumerate(batch_ranges):
    batch_df = df[start:end]
    print(f"\n=== Batch {idx + 1} (rows {start}:{end}) ===")
    

    source_schema = batch_df['source'].tolist()
    target_schema = batch_df['target'].tolist()

    # Load precomputed embeddings
    source_embeds = np.load(f"batch{idx + 1}_source_embeds.npy")
    target_embeds = np.load(f"batch{idx + 1}_target_embeds.npy")

    # Compute similarity
    similarity_matrix = cosine_similarity(source_embeds, target_embeds)

    # -----------------------------------------------------------
# Graph Propagation (safe version)
# -----------------------------------------------------------
threshold = 0.72
alpha = 0.3
propagated_source_embeddings = []

for i, s_node in enumerate(source_schema):
    # Select neighbors above similarity threshold
    neighbors = [(target_embeds[j], similarity_matrix[i][j])
                 for j in range(len(target_schema)) if similarity_matrix[i][j] > threshold]

    if not neighbors:
        # No neighbors above threshold → keep original embedding
        propagated_source_embeddings.append(source_embeds[i])
        continue

    # Safely create neighbor embedding array
    neighbor_embeds = np.array([e for (e, _) in neighbors])
    weights = np.array([w for (_, w) in neighbors])

    # Ensure neighbor embeddings have correct shape
    if neighbor_embeds.ndim != 2 or neighbor_embeds.shape[1] != source_embeds.shape[1]:
        raise ValueError(f"Embedding dimension mismatch at source index {i}!")

    # Weighted average of neighbors
    weighted_avg = np.average(neighbor_embeds, axis=0, weights=weights)

    # Blend with original embedding
    new_embed = (1 - alpha) * source_embeds[i] + alpha * weighted_avg
    propagated_source_embeddings.append(new_embed)

propagated_source_embeddings = np.vstack(propagated_source_embeddings)
propagated_similarity_matrix = cosine_similarity(propagated_source_embeddings, target_embeds)
print("✅ Propagation complete. Similarities updated.")


# Derive top-3 auto-mappings
auto_mapping = {}
top_k = 3

for i, s_node in enumerate(source_schema):
    sim_scores = [(target_schema[j], propagated_similarity_matrix[i][j]) for j in range(len(target_schema))]
    top_matches = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_k]
    best_target, best_score = top_matches[0]
    auto_mapping[s_node] = best_target

    # Collect top-1 predictions for metrics
    all_top1_preds.append(best_target)
    all_true_targets.append(target_schema[i])

    print(f"{s_node} → {best_target} (score: {best_score:.3f})")

    all_auto_mappings[f"batch_{idx + 1}"] = auto_mapping

   
# ---------------------------------------------------------------
# Compute overall metrics
# ---------------------------------------------------------------
accuracy = accuracy_score(all_true_targets, all_top1_preds)
precision = precision_score(all_true_targets, all_top1_preds, average='weighted', zero_division=0)
recall = recall_score(all_true_targets, all_top1_preds, average='weighted', zero_division=0)
f1 = f1_score(all_true_targets, all_top1_preds, average='weighted', zero_division=0)

# Skip rate = fraction of top-1 predictions that do NOT match the true target
skip_rate = 1 - accuracy

print("\n✅ Evaluation Metrics Across All Batches:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"Skip Rate  : {skip_rate:.4f}")

# ---------------------------------------------------------------
# Print all batch mappings
# ---------------------------------------------------------------
print("\n✅ Final dictionary of propagated mappings for all batches:")
for batch_name, mapping in all_auto_mappings.items():
    print(f"\n{batch_name}:")
    for s, t in mapping.items():
        print(f"  {s} → {t}")

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
