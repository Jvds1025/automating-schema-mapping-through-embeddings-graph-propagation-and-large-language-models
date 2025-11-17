df1 = # Add your dataset here
mapping1 = # Load the output JSON here
import pandas as pd

# possibly restrict your df to a specific amount of rows (batch)
df1 = # df[0:50] for example

df_map = pd.DataFrame(mapping1)
df_map.rename(columns={"output": "predicted_target"}, inplace=True)

# merge predictions
df_eval = df1.merge(df_map, on="source", how="left").drop_duplicates(subset=df1.columns)

# convert prediction into a list (so it works for top-k eval)
df_eval["pred_list"] = df_eval["predicted_target"].apply(
    lambda x: [] if x in [None, "None"] else [x]
)


TP1 = FP1 = FN1 = TN1 = 0
TP3 = FP3 = FN3 = TN3 = 0

for _, row in df_eval.iterrows():
    label = row["label"]
    true_target = row["target"]
    preds = row["pred_list"]

    top1 = preds[:1]

    # TOP 1
    if label == 1:
        if true_target in top1:
            TP1 += 1
        else:
            FN1 += 1
    else:
        if len(top1) > 0:
            FP1 += 1
        else:
            TN1 += 1

def metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2*precision*recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn)
    return acc, precision, recall, f1

print("TOP-1:", metrics(TP1, FP1, FN1, TN1))

print(TP1, FP1, FN1, TN1)

df_eval

