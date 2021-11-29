import json
import csv
import pathlib
from faithfulness.SRL import SRL
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.utils.correlation import pearson, spearman

# Load input data
with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["summary_sentences"], x["source_sentences"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "srl_em"
save_path = str(pathlib.Path().resolve()) + f"/{metric_name}"
metric = SRL(metric=ExactMatch, batch_mode=True, save_path=save_path, model_path="/home/tim/Development/faithfulness/models/structured-prediction-srl-bert.2020.12.15.tar.gz")

# Calculate faithfulness
precisions, recalls, f1s, similarities, alignments, summaries_phrases, sources_phrases = metric.score_batch(summaries, sources, True).values()

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_precision"] = precisions[idx]
    x[f"{metric_name}_recall"] = recalls[idx]
    x[f"{metric_name}_f1"] = f1s[idx]
    x[f"{metric_name}_similarities"] = similarities[idx]
    x[f"{metric_name}_alignment"] = similarities[idx]
    x[f"{metric_name}_summary_phrases"] = summaries_phrases[idx]
    x[f"{metric_name}_source_phrases"] = sources_phrases[idx]
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
