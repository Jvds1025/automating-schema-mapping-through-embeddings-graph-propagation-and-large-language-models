from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from peft import PeftModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os 
import psutil

start_time = time.time()

# -----------------------------------------------------------
# Utility: Memory usage tracker
def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / 1e9
    gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_mem_gb = cpu_mem_gb + gpu_mem_gb
    print(f"[Memory] {step_name}: CPU={cpu_mem_gb:.2f} GB | GPU={gpu_mem_gb:.2f} GB | Total={total_mem_gb:.2f} GB")

# -----------------------------------------------------------
# Load model & tokenizer
tokenizer_embedding = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
model_embedding = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model_path = "best_model.pt"

print_memory_usage("After loading model/tokenizer")

# -----------------------------------------------------------
# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="test_augmented.jsonl")["train"]
df = pd.DataFrame(dataset)
df = df[300:350]  # optional subset for faster testing

# -----------------------------------------------------------
# Load LoRA fine-tuned weights
try:
    model = PeftModel.from_pretrained(model_embedding, lora_model_path)
    print("✅ Loaded LoRA adapter via PeftModel.")
except Exception:
    print("⚠️ Falling back to manual state_dict load...")
    state_dict = torch.load(lora_model_path, map_location=device)
    model_embedding.load_state_dict(state_dict, strict=False)
    model = model_embedding

model.to(device)
model.eval()

print_memory_usage("After loading dataset")

# -----------------------------------------------------------
# Compute embeddings
def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer_embedding(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)

print("Computing source embeddings...")
source_embeddings = get_embeddings(df["source"].tolist())

print("Computing target embeddings...")
target_embeddings = get_embeddings(df["target"].tolist())

np.save("source_embeddings.npy", source_embeddings)
np.save("target_embeddings.npy", target_embeddings)
print("✅ Embeddings saved successfully.")

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")

print_memory_usage("After applying LoRA weights")

# -----------------------------------------------------------
# Reload embeddings (safety)
source_embeds = np.load("source_embeddings.npy")
target_embeds = np.load("target_embeddings.npy")

source_schema = df['source'].tolist()
target_schema = df['target'].tolist()

threshold = 0.72
top_k = 3
similarity_matrix = cosine_similarity(source_embeds, target_embeds)
print_memory_usage("After computing similarity matrix")

# -----------------------------------------------------------
# >>> NEW SECTION: Full Graph Propagation with Endpoint Awareness + Softmax Weighting <<<
print("\n🔁 Performing full graph propagation with endpoint awareness and softmax weighting...")

alpha = 0.3  # neighbor similarity influence
beta = 0.2   # same-endpoint structural influence
temperature = 0.1  # controls sharpness of softmax weighting (lower = more selective)

# Extract endpoint (table) names
def get_endpoint(col_name):
    return col_name.split('.')[0] if '.' in col_name else col_name

source_endpoints = [get_endpoint(s) for s in source_schema]
target_endpoints = [get_endpoint(t) for t in target_schema]

propagated_source_embeddings = []

for i, s_node in enumerate(source_schema):
    s_endpoint = source_endpoints[i]

    # 🧠 1️⃣ Neighbor propagation using all targets (softmax-weighted)
    sim_scores = similarity_matrix[i]
    exp_weights = np.exp(sim_scores / temperature)
    weights = exp_weights / (np.sum(exp_weights) + 1e-8)  # softmax normalization
    weighted_avg = np.average(target_embeds, axis=0, weights=weights)

    # 🧩 2️⃣ Endpoint structural propagation (context from same table or data connector)
    same_endpoint_indices = [k for k, ep in enumerate(source_endpoints) if ep == s_endpoint and k != i]
    if same_endpoint_indices:
        same_ep_embeds = np.stack([source_embeds[k] for k in same_endpoint_indices])
        endpoint_context = np.mean(same_ep_embeds, axis=0)
    else:
        endpoint_context = np.zeros_like(source_embeds[i])

    # ⚖️ 3️⃣ Blend components: original + neighbor (semantic) + endpoint (structural)
    new_embed = (
        (1 - alpha - beta) * source_embeds[i]
        + alpha * weighted_avg
        + beta * endpoint_context
    )

    propagated_source_embeddings.append(new_embed)

propagated_source_embeddings = np.vstack(propagated_source_embeddings)

# 🔄 Recompute similarities after propagation
propagated_similarity_matrix = cosine_similarity(propagated_source_embeddings, target_embeds)
print("✅ Full graph propagation with softmax weighting complete. Similarity matrix updated.")
print_memory_usage("After full softmax propagation")

# -----------------------------------------------------------
# Build graph with propagated similarities
G = nx.DiGraph()
for node in source_schema + target_schema:
    G.add_node(node)

for i, s_node in enumerate(source_schema):
    sim_scores = [(target_schema[j], propagated_similarity_matrix[i][j]) for j in range(len(target_schema))]
    above_threshold = [t for t in sim_scores if t[1] > threshold]
    top3 = sorted(above_threshold, key=lambda x: x[1], reverse=True)[:top_k]
    for t_node, score in top3:
        G.add_edge(s_node, t_node, weight=score)

# -----------------------------------------------------------
# Manual user validation
def offer_user_choice(source_col, options):
    format_option_text = '\n'.join([f"{i+1}. {val}" for i, val in enumerate(options)])
    choice = input(f"Enter the correct destination mapping for {source_col}:\n{format_option_text}\n")
    return source_col, options[int(choice) - 1]

print("\nCandidate Schema Mappings (after propagation):")
cur_source_node = "first"
current_dest_options = []
mapping = {}

for u, v, data in G.edges(data=True):
    if cur_source_node == "first":
        cur_source_node = u
    if cur_source_node != u:
        current_dest_options.append("None of the above")
        source_col, target_col = offer_user_choice(cur_source_node, current_dest_options)
        mapping[source_col] = target_col
        current_dest_options = []
        cur_source_node = u
        print("\nNew mapping item:")

    print(f"{u} -> {v} (similarity: {data['weight']:.2f})")
    current_dest_options.append(v)

source_col, target_col = offer_user_choice(cur_source_node, current_dest_options)
mapping[source_col] = target_col

print(f"\n✅ Final mapping created: {str(mapping)}")
print_memory_usage("End of script")

# ✅ EVALUATION SECTION (after user selections)

# Convert user mapping to DataFrame
df_batch = pd.DataFrame(list(mapping.items()), columns=['source', 'predicted_target_field'])

# Merge with original labeled dataset
df_merged8 = pd.merge(df, df_batch, on='source', how='left')

# Evaluate metrics
df_merged8["pred_correct"] = (df_merged8["predicted_target_field"] == df_merged8["target"]).astype(int)

total = len(df_merged8)
correct = (df_merged8["pred_correct"] & (df_merged8["label"] == 1)).sum()
true_positives = correct
false_positives = ((df_merged8["pred_correct"] == 1) & (df_merged8["label"] == 0)).sum()
false_negatives = ((df_merged8["pred_correct"] == 0) & (df_merged8["label"] == 1)).sum()
true_negatives = ((df_merged8["pred_correct"] == 0) & (df_merged8["label"] == 0)).sum()

precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (true_positives + false_negatives + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
accuracy = (true_positives + true_negatives) / total

# Handle NaN predictions
Nan_correct = ((df_merged8["pred_correct"] == 0) & (df_merged8["predicted_target_field"].isna())).sum()
Nan_total = df_merged8["predicted_target_field"].isna().sum()
null_total = (df_merged8["pred_correct"] == 0).sum()
Nan_division = (Nan_correct / null_total) if Nan_total > 0 else 0.0

print("\n📊 Evaluation Metrics")
print(f"Accuracy:       {accuracy:.3f}")
print(f"Precision:      {precision:.3f}")
print(f"Recall:         {recall:.3f}")
print(f"F1 Score:       {f1:.3f}")
print(f"NaN Division:   {Nan_division:.3f}")
