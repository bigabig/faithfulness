import json
import csv
import pathlib

# Load input data
from faithfulness.QGQA import QGQA
from faithfulness.similarity.F1 import F1
from faithfulness.utils.correlation import pearson, spearman

with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["summary"], x["source"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "qgqa_f1"
similarity_metric = F1()
save_path = str(pathlib.Path().resolve()) + f"/{metric_name}"
metric = QGQA(metric=similarity_metric, batch_mode=True, save_path=save_path)

# Calculate faithfulness
scores, questions = metric.score_batch(summaries, sources, True).values()

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_score"] = scores[idx]
    x[f"{metric_name}_questions"] = questions[idx]
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
