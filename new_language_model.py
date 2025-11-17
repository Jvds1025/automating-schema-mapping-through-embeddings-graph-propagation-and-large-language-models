import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import psutil
import os 

start_time = time.time()
total_tokens = 0 

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
df = df[200:250]

def split_field(field_name):
    """Split 'endpoint.column' into endpoint (table) and column."""
    parts = field_name.split('.')
    if len(parts) == 2:
        return parts[0], parts[1]

df[['src_endpoint', 'src_column']] = df['source'].apply(lambda x: pd.Series(split_field(x)))
df[['tgt_endpoint', 'tgt_column']] = df['target'].apply(lambda x: pd.Series(split_field(x)))

#Create embeddings
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

# ____________________________________________
top_k = 5
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

# Create prediction dictionary
pred_dict = {item['source']: [t for t, s in item['top_targets']] for item in filtered_candidates}
print("Dictionary created.")

# ___________________________________________________________________________
# LLM Reasoning here!!!!!!!!!!!!!!!!!!!!!!!!!

import json
from transformers import AutoModelForCausalLM

# Load generative model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
prompt_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype="auto",
    device_map="auto"
)

# Build candidate data for LLM input
llm_candidates = []
for item in filtered_candidates:
    src = item["source"]
    candidates = [{"target": t, "score": float(s)} for t, s in item["top_targets"]]
    
    # Always add 'None' as an option
    candidates.append({"target": "None", "score": None})
    
    llm_candidates.append({
        "source": src,
        "candidates": candidates
    })


def build_mapping_prompt(candidate_list, few_shot_examples):
    prompt = (
    "You are an expert data mapping assistant.\n\n"
    "Your task is to select **one target** from the given candidates for each source field.\n\n"
    "Rules:\n"
    "1. Choose the target that best matches the source semantically and structurally.\n"
    "2. Consider both the endpoint (table) and column names when determining the match.\n"
    "3. Similarity scores are provided for reference but do NOT blindly pick the highest score.\n"
    "4. If the candidate does not match the source field in meaing (even if embedding similairty is high),select **None**.\n"
    "5. Respond ONLY with a single JSON array of objects in this format:\n\n"
    "[\n"
    '  {"source": "source_field_1", "output": "best_target_1"},\n'
    '  {"source": "source_field_2", "output": "best_target_2"},\n'
    '  {"source": "...", "output": "..."}\n'
    "]\n\n"
    "Do not add explanations or any extra text. Include all source fields from the input candidate list."
)

    
    # Append candidates as JSON
    prompt += json.dumps(candidate_list, indent=2)
    
    return prompt

    

prompt_text = build_mapping_prompt(llm_candidates, few_shot_examples=[
    {"source": "user_id", "target": "id"},
        
    # ✅ Example where a clear match exists
    {"source": "Invoice.Amount", "target": "Invoices.Total"},
    {
            "source": "customer_profile_picture",
            "candidates": [
                {"target": "orders.total_price", "score": 0.72},
                {"target": "orders.order_date", "score": 0.69}
            ],
            "expected_output": "None"
        }
    ]
)

inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(device)
num_input_tokens = inputs["input_ids"].numel()

with torch.no_grad():
    outputs = prompt_model.generate(**inputs, max_new_tokens=1500)
    
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_text)

output_tokens = tokenizer(response_text, return_tensors="pt")
num_output_tokens = output_tokens['input_ids'].numel()

# Extract and parse the output
import re
try:
    match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if match:
        llm_mapping = json.loads(match.group(1))
    else:
        llm_mapping = json.loads(response_text)
except json.JSONDecodeError:
    print("⚠️ Could not parse JSON from LLM output.")
    llm_mapping = {}

print("\n✅ Final refined mappings:")
for k, v in llm_mapping.items():
    print(f"{k} → {v}")

end_time = time.time()
total_time = end_time - start_time

print(f"Total elapsed time: {total_time:.2f} seconds")
print_peak_memory_usage("End of script")

print("\n===== TOKEN USAGE REPORT =====")
print(f"Embedding model tokens: {total_tokens}")
print(f"LLM input tokens:         {num_input_tokens}")
print(f"LLM output tokens:        {num_output_tokens}")
print(f"TOTAL LLM tokens:         {num_input_tokens + num_output_tokens}")
print(f"GRAND TOTAL tokens:       {total_tokens + num_input_tokens + num_output_tokens}")

