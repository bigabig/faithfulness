import json
import csv
from faithfulness.SentSim import SentSim
from faithfulness.similarity.SentCos import SentCos
from faithfulness.utils.correlation import pearson, spearman

# Load input data
with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries_sentences, sources_sentences, faithfulness_scores = zip(*[(x["summary_sentences"], x["source_sentences"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "sentsim_sentcos_all-mpnet-base-v2"
similarity_metric = SentCos()
metric = SentSim(metric=similarity_metric)

# Calculate faithfulness
precisions, recalls, f1s, alignments, similarities = metric.score_batch(summaries_sentences, sources_sentences, True).values()


# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_precision"] = precisions[idx]
    x[f"{metric_name}_recall"] = recalls[idx]
    x[f"{metric_name}_f1"] = f1s[idx]
    x[f"{metric_name}_alignment"] = alignments[idx]
    x[f"{metric_name}_similarity"] = similarities[idx]
with open(f"{metric_name}.json", "w", encoding="UTF-8") as outfile:
    json.dump(data, outfile)

# Calculate correlation
table = [["Method", "Pearson", "Spearman"]]
for scores, label in zip([precisions, recalls, f1s], ["_precision", "_recall", "_f1"]):
    table.append([metric_name + label, pearson(scores, faithfulness_scores), spearman(scores, faithfulness_scores)])

# Save correlation as csv file
with open(f"{metric_name}.csv", "w", encoding="UTF-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(table)

print("Finished! :)")
