from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch

train_augmented = load_dataset("json", data_files="train_augmented.jsonl")["train"]
val_augmented = load_dataset("json", data_files="val_augmented.jsonl")["train"]
test_augmented = load_dataset("json", data_files="test_augmented.jsonl")["train"]

#Load this on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-4B",
    device_map="auto",          # Let transformers decide which parts fit on GPU
    dtype=torch.float16,  # use half precision to save memory
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
model.to(device)

#-----------------------------------------------------------------------------

class SchemaPairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_text = pair['source']  # Access source using index 0
        tgt_text = pair['target']  # Access target using index 1
        label = pair['label']   # Access label using index 2

        # Tokenize source and target
        src_enc = self.tokenizer(src_text, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
        tgt_enc = self.tokenizer(tgt_text, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids_src": src_enc["input_ids"].squeeze(0),
            "attention_mask_src": src_enc["attention_mask"].squeeze(0),
            "input_ids_tgt": tgt_enc["input_ids"].squeeze(0),
            "attention_mask_tgt": tgt_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(label), dtype=torch.float)
        }

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # for embeddings
    r=8, # can be lower to at least 4
    lora_alpha=32, # can become a lot lower, test on 16
    lora_dropout=0.1, # Test dropout 0.2
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

train_dataset = SchemaPairsDataset(train_augmented, tokenizer) #train_df_load for old train df
val_dataset = SchemaPairsDataset(val_augmented, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


def cosine_loss(batch):
    source_emb = model(input_ids=batch["input_ids_src"], attention_mask=batch["attention_mask_src"]).last_hidden_state[:,0,:]
    target_emb = model(input_ids=batch["input_ids_tgt"], attention_mask=batch["attention_mask_tgt"]).last_hidden_state[:,0,:]

    cos_sim = F.cosine_similarity(source_emb, target_emb)
    labels = batch["label"]
    loss = (labels * (1 - cos_sim) + (1 - labels) * cos_sim).mean()
    return loss

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 20
patience = 2
best_val_loss = float('inf')
epochs_no_improve = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(epochs):

    # ===== Training =====
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        loss = cosine_loss(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # ===== Validation =====
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            loss = cosine_loss(batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ===== Early stopping check =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Optionally save the best model
        torch.save(model.state_dict(), "best_model2.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break





