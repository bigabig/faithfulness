import json
import csv
from faithfulness.OpenIE import OpenIE
from faithfulness.similarity.SentCos import SentCos
from faithfulness.utils.correlation import pearson, spearman
import pathlib

# Load input data
with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
data = data[:10]
summaries, sources, faithfulness_scores = zip(*[(x["summary"], x["source"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "openie_sentcos_test"
save_path = str(pathlib.Path().resolve()) + f"/{metric_name}"
metric = OpenIE(metric=SentCos, batch_mode=True, save_path=save_path)

# Calculate faithfulness
precisions, recalls, f1s, alignments, similarities, summaries_triples, sources_triples, summaries_sentences, sources_sentences = metric.score_batch(summaries, sources, True).values()

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_precision"] = precisions[idx]
    x[f"{metric_name}_recall"] = recalls[idx]
    x[f"{metric_name}_f1"] = f1s[idx]
    x[f"{metric_name}_alignment"] = alignments[idx]
    x[f"{metric_name}_similarity"] = similarities[idx]
    x[f"{metric_name}_summary_triples"] = summaries_triples[idx]
    x[f"{metric_name}_source_triples"] = sources_triples[idx]
    x[f"{metric_name}_summary_sentences"] = summaries_sentences[idx]
    x[f"{metric_name}_source_sentences"] = sources_sentences[idx]
with open(f"{metric_name}.json", "w", encoding="UTF-8") as outfile:
    json.dump(data, outfile, default=lambda x: x.__dict__)

# Calculate correlation
table = [["Method", "Pearson", "Spearman"]]
for scores, label in zip([precisions, recalls, f1s], ["_precision", "_recall", "_f1"]):
    table.append([metric_name + label, pearson(scores, faithfulness_scores), spearman(scores, faithfulness_scores)])

# Save correlation as csv file
with open(f"{metric_name}.csv", "w", encoding="UTF-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(table)

print("Finished! :)")
