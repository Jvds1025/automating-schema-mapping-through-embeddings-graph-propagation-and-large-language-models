import pandas as pd

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
