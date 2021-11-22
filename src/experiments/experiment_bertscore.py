import json
import csv
from faithfulness.BERTScore import BERTScore, BERTScoreMethod

# Load input data
from faithfulness.utils.correlation import pearson, spearman

with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["source_sentences"], x["summary_sentences"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "bertscore_sent"
bs = BERTScore(method=BERTScoreMethod.SENT)

# Calculate faithfulness
precisions, recalls, f1s, similarities, summaries_tokens, sources_tokens = bs.score_batch(summaries, sources, True).values()  # [[precision, recall, f1], ...]

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_precision"] = precisions[idx]
    x[f"{metric_name}_recall"] = recalls[idx]
    x[f"{metric_name}_f1"] = f1s[idx]
    x[f"{metric_name}_similarities"] = similarities[idx]
    x[f"{metric_name}_summary_tokens"] = summaries_tokens[idx]
    x[f"{metric_name}_source_tokens"] = sources_tokens[idx]
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
