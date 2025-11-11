import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import re # Added for robust JSON parsing
from datasets import load_dataset
from peft import PeftModel
import time

start_time = time.time()


#  1. GLOBAL VARIABLE DEFINITIONS (Fixes NameError: OUTPUT_FILE) 
OUTPUT_FILE = "final_mapping_result.json"


model_embedding = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map=None,          # Let transformers decide which parts fit on GPU
    dtype=torch.float16,  # use half precision to save memory
    low_cpu_mem_usage=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
model_embedding.to(device)

lora_model_path = "best_model.pt"            # or folder if you saved via Peft save_pretrained()

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="test_augmented.jsonl")["train"]
df = pd.DataFrame(dataset)
df = df[0:50]

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
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
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
print("✅ Embeddings saved successfully: source_embeddings.npy and target_embeddings.npy")

#  2. MODEL SETUP 
# NOTE: The Qwen model is very large (4B) and requires significant GPU memory.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
prompt_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype="auto",
    device_map="auto"
)
# NOTE: Assuming "Saved_Embeddings/*.npy" exist and load correctly.
try:
    source_embeds = np.load("source_embeddings.npy")
    target_embeds = np.load("target_embeddings.npy")
except FileNotFoundError:
    print("WARNING: Embeddings files not found. Creating mock embeddings.")

# Compute similarity matrix
# NOTE: Ensure source_embeds and target_embeds have at least 15 rows for slicing
if source_embeds.shape[0] < 15 or target_embeds.shape[0] < 15:
     print("Error: Not enough embeddings for slicing (using all available).")
     source_embeds = source_embeds
     target_embeds = target_embeds
     
     
source_columns = df['source'].tolist()
target_columns = df['target'].tolist()
     
similarity_matrix = cosine_similarity(source_embeds[:len(source_columns)], target_embeds[:len(target_columns)])

# Parameters
top_k = 3
threshold = 0.7

# Prepare candidate mappings for prompting
candidate_mappings = []

for i, src in enumerate(source_columns):
    # Skip if we run out of similarity matrix rows (due to slicing errors)
    if i >= similarity_matrix.shape[0]:
        continue 
        
    sims = [(target_columns[j], float(similarity_matrix[i][j])) for j in range(len(target_columns))]
    top_candidates = [t for t in sims if t[1] >= threshold]
    
    # keep only top_k
    top_candidates = sorted(top_candidates, key=lambda x: x[1], reverse=True)[:top_k]
    
    # append to candidate list
    candidate_mappings.append({
        "source": src,
        "candidates": [{"target": t[0], "score": t[1]} for t in top_candidates]
    })

# 3. Prompt Construction
prompt_input_data = candidate_mappings

# --------------------------------------------------------------------------
def build_prompt_from_candidates(prompt_input_data, few_shot_examples= 
                                 [
    {"Journal.TaxCode": "Customers.CustomerID"},
    {"Amount": "Invoices.Total"}
]):
    """
    Build a text prompt for LLM from candidate mappings.
    
    Args:
        prompt_input_data (list of dict): Each dict has "source" and "candidates" keys.
    
    Returns:
        str: Prompt text ready for Qwen.
    """
    
    prompt = (
        "You are a data mapping assistant. For each source column, select exactly ONE target column from the candidates.\n"
        "Only output a JSON mapping of {source_field: target_field}.\n"
        "Do not include multiple candidates per source.\n\n"
    )

    # Add the candidate mappings
    prompt += "Candidates:\n"
    # Dumps the actual data structure into the prompt string
    prompt += json.dumps(prompt_input_data, indent=2) 

    return prompt

prompt_text = build_prompt_from_candidates(prompt_input_data, few_shot_examples=[
    {"source": "user_id", "target": "id"}
])

# --------------------------------------------------------------------------
def query_qwen(prompt_text, model_name="Qwen/Qwen3-4B-Instruct-2507"):
    """
    Query the Qwen 3-4B model with the provided prompt.
    NOTE: This is the actual execution against the Hugging Face model.
    """

    # Tokenize the input prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate a response
    with torch.no_grad():
        outputs = prompt_model.generate(**inputs, max_length=53000, num_return_sequences=1)

    # Decode and return the response (raw string)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. EXECUTION AND JSON PARSING (Fixes NameError and prepares string for saving) ---
final_mapping_string = query_qwen(prompt_text)

# Initialize final_mapping as None
final_mapping = None

if final_mapping_string:
    try:
        match = re.search(r"(\{.*\})", final_mapping_string, re.DOTALL)
        if match:
            # We found the JSON object as a string
            json_data = match.group(0)
            # Load the string into a Python dictionary
            final_mapping = json.loads(json_data)
        else:
            # If no JSON block is found, try to load the whole string (less reliable)
            final_mapping = json.loads(final_mapping_string)
            
    except json.JSONDecodeError as e:
        print(f"\nERROR: Failed to parse LLM output as JSON: {e}")
        print(f"Raw LLM Output (could not parse): {final_mapping_string}")
        final_mapping = None # Ensure it is None if parsing fails

# 5. PRINTING AND SAVING 
if final_mapping is not None:
    print("\n--- FINAL MAPPING RESULT (Dictionary) ---")
    print("Final Mapping:", final_mapping)
    
    # Save the result to a JSON file
    try:
        # OUTPUT_FILE is now defined globally at the top
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_mapping, f, indent=4)
        print(f"\nSuccessfully saved mapping result to file: {OUTPUT_FILE}")
    except Exception as e:
        # This block catches file write errors
        print(f"\nERROR: Could not save mapping to file {OUTPUT_FILE}: {e}")

lm_predictions = {}
for line in final_mapping_string.strip().split("\n"):
    source, target = line.split(";")
    lm_predictions[source.strip()] = target.strip()


end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")