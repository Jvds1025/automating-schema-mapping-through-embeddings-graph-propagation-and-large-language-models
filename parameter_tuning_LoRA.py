from torch.utils.data import Dataset, DataLoader
import torch
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from itertools import product
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_augmented = pd.read_csv("trainingset_.csv")
val_augmented = pd.read_csv("validationset_.csv")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")

class SchemaPairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        src_text = str(pair['source'])  
        tgt_text = str(pair['target'])  
        label = pair['label']

        src_enc = self.tokenizer(
            src_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids_src": src_enc["input_ids"].squeeze(0),
            "attention_mask_src": src_enc["attention_mask"].squeeze(0),
            "input_ids_tgt": tgt_enc["input_ids"].squeeze(0),
            "attention_mask_tgt": tgt_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(label), dtype=torch.float)
        }

# Loss function
def cosine_loss(batch, model):
    source_emb = model(
        input_ids=batch["input_ids_src"],
        attention_mask=batch["attention_mask_src"]
    ).last_hidden_state[:,0,:]

    target_emb = model(
        input_ids=batch["input_ids_tgt"],
        attention_mask=batch["attention_mask_tgt"]
    ).last_hidden_state[:,0,:]

    cos_sim = F.cosine_similarity(source_emb, target_emb)
    labels = batch["label"]
    loss = (labels * (1 - cos_sim) + (1 - labels) * cos_sim).mean()
    return loss

param_grid = {
    "r": [4,8], 
    "alpha": [16, 32],
    "dropout": [0.0, 0.1],
    "lr": [5e-6, 1e-5],
    "batch_size": [8],
    "max_length": [64]
}

def expand_grid(grid):
    keys = grid.keys()
    for values in product(*grid.values()):
        yield dict(zip(keys, values))

configs = list(expand_grid(param_grid))
print("Total configs to try:", len(configs))

# Hyperparamter tuning
results = []

for cfg in configs:
    print("\n Running config:", cfg)

    # Load fresh base model
    base_model = AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-4B",
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    # DataLoaders
    train_dataset = SchemaPairsDataset(train_augmented, tokenizer, max_length=cfg["max_length"])
    val_dataset = SchemaPairsDataset(val_augmented, tokenizer, max_length=cfg["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    best_val_loss = float("inf")

    # Quick training for hyperparameter tuning (2 epochs)
    for epoch in range(8):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            optimizer.zero_grad()
            loss = cosine_loss(batch, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k,v in batch.items()}
                val_loss += cosine_loss(batch, model).item()
        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)

    print(f"Config finished. Best Val Loss: {best_val_loss:.4f}")
    results.append({**cfg, "val_loss": best_val_loss}) 

# Results
results = sorted(results, key=lambda x: x["val_loss"])
print("\nBEST CONFIG")
print(results[0])
