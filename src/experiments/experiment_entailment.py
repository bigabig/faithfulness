import json
import csv
from faithfulness.Entailment import Entailment, EntailmentMethod
from faithfulness.utils.correlation import pearson, spearman

# 6:25 min

# Load input data
with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["summary"], x["source"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "entailment_doc"
method = EntailmentMethod.DOC
entailment = Entailment(method=method, max_length=512)

# Calculate faithfulness
if method == EntailmentMethod.SENT:
    scores, alignments, entailments = entailment.score_batch(summaries, sources, True).values()
elif method == EntailmentMethod.DOC:
    scores = entailment.score_batch(summaries, sources, True)['score']

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}"] = scores[idx]
    if method == EntailmentMethod.SENT:
        x[f"{metric_name}_alignment"] = alignments[idx]
        x[f"{metric_name}_entailment"] = entailments[idx]
with open(f"{metric_name}.json", "w", encoding="UTF-8") as outfile:
    json.dump(data, outfile)

# Calculate correlation
table = [["Method", "Pearson", "Spearman"]]
for scores, label in zip([scores], ["_score"]):
    table.append([metric_name + label, pearson(scores, faithfulness_scores), spearman(scores, faithfulness_scores)])

# Save correlation as csv file
with open(f"{metric_name}.csv", "w", encoding="UTF-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(table)

print("Finished! :)")
