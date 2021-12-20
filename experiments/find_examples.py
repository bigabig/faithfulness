import json
import random
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
difference = torch.tensor(scores - faithfulness)

# find top k best and worst examples
under_example_ids = torch.topk(difference, k=200, largest=False).indices.tolist()
over_example_ids = torch.topk(difference, k=200, largest=True).indices.tolist()

random_example_ids = set()
while len(random_example_ids) < 50:
    random_example_ids.add(random.randrange(0, len(data)))

# dont take the 50 worst or 50 best
under_example_ids = under_example_ids[150:]
over_example_ids = over_example_ids[150:]

under_examples = [data[idx] for idx in under_example_ids]
over_examples = [data[idx] for idx in over_example_ids]
random_examples = [data[idx] for idx in random_example_ids]

# Write data
out_file_under = in_file.with_name(in_file.name.replace(".json", "_examples_under.json"))
out_file_over = in_file.with_name(in_file.name.replace(".json", "_examples_over.json"))
out_file_random = in_file.with_name(in_file.name.replace(".json", "_examples_random.json"))
with out_file_under.open(encoding="UTF-8", mode="w") as file:
    json.dump(under_examples, file)
with out_file_over.open(encoding="UTF-8", mode="w") as file:
    json.dump(over_examples, file)
with out_file_random.open(encoding="UTF-8", mode="w") as file:
    json.dump(random_examples, file)
