import json
from pathlib import Path
import numpy as np
import torch

dataset = "xsum"
metric = "entailment"
variant = "sent"
in_file = Path(f"{dataset}/{metric}/{metric}_{variant}.json")
max_num_sentences = 30
score_key = "precision"  # or recall or f1

with in_file.open(mode="r", encoding="UTF-8") as file:
    data = json.load(file)

# filter data by sentence length
print(len(data))
data = [x for x in data if len(x['source_sentences']) <= max_num_sentences]
print(len(data))

# calculate difference between human faithfulness & metric faithfulness judgements
scores = np.array([x[score_key] for x in data])
faithfulness = np.array([x["faithfulness"] for x in data])
difference = torch.tensor(np.abs(scores - faithfulness))

# find top k best and worst examples
good_example_ids = torch.topk(difference, k=100, largest=False).indices.tolist()
bad_example_ids = torch.topk(difference, k=100, largest=True).indices.tolist()

# dont take the 50 worst or 50 best
good_example_ids = good_example_ids[50:]
bad_example_ids = bad_example_ids[50:]

good_examples = [data[idx] for idx in good_example_ids]
bad_examples = [data[idx] for idx in bad_example_ids]

# Write data
out_file_good = in_file.with_name(in_file.name.replace(".json", "_examples_good.json"))
out_file_bad = in_file.with_name(in_file.name.replace(".json", "_examples_bad.json"))
with out_file_good.open(encoding="UTF-8", mode="w") as file:
    json.dump(good_examples, file)
with out_file_bad.open(encoding="UTF-8", mode="w") as file:
    json.dump(bad_examples, file)
