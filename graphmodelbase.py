from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from peft import PeftModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import psutil
import os



start_time = time.time()

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / 1e9
    gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total_mem_gb = cpu_mem_gb + gpu_mem_gb
    print(f"[Memory] {step_name}: CPU={cpu_mem_gb:.2f} GB | GPU={gpu_mem_gb:.2f} GB | Total={total_mem_gb:.2f} GB")

tokenizer_embedding = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
model_embedding = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",          # Let transformers decide which parts fit on GPU
    dtype=torch.float16,  # use half precision to save memory
    low_cpu_mem_usage=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
lora_model_path = "best_model.pt"
# ________________________________________________________________________________
# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="test_augmented.jsonl")["train"]
df = pd.DataFrame(dataset)
df = df[350:] # Limit to first 50 rows for faster testing


# Apply LoRA fine-tuned weights
try:
    # Case 1: if best_model.pt is a LoRA adapter directory (preferred)
    model = PeftModel.from_pretrained(model_embedding, lora_model_path)
    print("Loaded LoRA adapter via PeftModel.")
except Exception:
    # Case 2: if best_model.pt is a raw state_dict (manual load)
    print("Falling back to manual state_dict load...")
    state_dict = torch.load(lora_model_path, map_location=device)
    model_embedding.load_state_dict(state_dict, strict=False)
    model = model_embedding

model.to(device)
model.eval()

def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer_embedding(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)
            # mean pooling across tokens
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    return np.vstack(all_embeddings)

# Compute embeddings
print("Computing source embeddings...")
source_embeddings = get_embeddings(df["source"].tolist())
print("Computing target embeddings...")
target_embeddings = get_embeddings(df["target"].tolist())

# Save embeddings
np.save("source_embeddings.npy", source_embeddings)
np.save("target_embeddings.npy", target_embeddings)
print("Embeddings saved successfully: source_embeddings.npy and target_embeddings.npy")

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")

# NOTE: Assuming "Saved_Embeddings/*.npy" exist and load correctly.
try:
    source_embeds = np.load("source_embeddings.npy")
    target_embeds = np.load("target_embeddings.npy")
except FileNotFoundError:
    print("WARNING: Embeddings files not found. Creating mock embeddings.")

source_schema = df['source'].tolist()
target_schema = df['target'].tolist()

threshold = 0.72  # similarity threshold for an edge
top_k = 3  # number of top mappings to consider
similarity_matrix = cosine_similarity(source_embeds, target_embeds)
# Create an undirected graph
G = nx.DiGraph() 
""" Test this also with a directed graph? such as nx.DiGraph() """
""" Create a graph with all strong mappings. Top three mappings per source node """
# Add nodes
for node in source_schema + target_schema:
    G.add_node(node)
for i, s_node in enumerate(source_schema):
    threshold = 0.72
    sim_scores = [(target_schema[j], similarity_matrix[i][j]) for j in range(len(target_schema))]
    above_threshold = [t for t in sim_scores if t[1] > threshold]
    top3 = sorted(above_threshold, key=lambda x: x[1], reverse=True)[:top_k]
    # add edges
    for t_node, score in top3:
        G.add_edge(s_node, t_node, weight=score)
        #TODO Potentially uncomment?
        # print(f"{s_node} -> {t_node} (similarity: {score:.2f})")

# ____________________________________________________________________________________________________________
def offer_user_choice(source_col, options):
    "Simple function that turns the choice of a user in the command line into a dictionary"
    format_option_text = '\n'.join([f"{i+1}. {val}" for i, val in enumerate(options)])
    # choice = int(input(f"Enter the correct destination mapping for {source_col}:\n{format_option_text}\n"))
    choice = input(f"Enter the correct destination mapping for {source_col}:\n{format_option_text}\n")
    # print(choice)
    # print(str(choice))
    return source_col, options[int(choice) - 1]
print("Candidate Schema Mappings (source -> target):")
 
cur_source_node = "first"
current_dest_options = []
mapping = {}

# Confident mappings
for u, v, data in G.edges(data=True):
    if cur_source_node == "first":
        cur_source_node = u

    if cur_source_node != u:
        current_dest_options.append("None of the above")
        source_col, target_col = offer_user_choice(cur_source_node, current_dest_options)
        mapping[source_col] = target_col
        # mapping.append(offer_user_choice(cur_source_node, current_dest_options))
        current_dest_options = [] 
        cur_source_node = u
        print("\nNew mapping item:") # New line to easily distinguish between options

    print(f"{u} -> {v} (similarity: {data['weight']:.2f})")
    current_dest_options.append(v)

source_col, target_col = offer_user_choice(cur_source_node, current_dest_options)
mapping[source_col] = target_col

print(f"\nFinal mapping that was created is: {str(mapping)}")
print_memory_usage("End of script")

