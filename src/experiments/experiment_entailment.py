import json
import csv
from faithfulness.Entailment import Entailment, EntailmentMethod
from faithfulness.utils.correlation import pearson, spearman

# 6:25 min

# Load input data
with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["summary_sentences"], x["source_sentences"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "entailment_sent_new2"
method = EntailmentMethod.SENT
entailment = Entailment(method=method, max_length=512)

# Calculate faithfulness
if method == EntailmentMethod.SENT:
    precisions, recalls, f1s, summary_alignments, summary_entailments, source_alignments, source_entailments = entailment.score_batch(summaries, sources, True).values()
elif method == EntailmentMethod.DOC:
    scores = entailment.score_batch(summaries, sources, True)['score']

# Save results as json file
for (idx, x) in enumerate(data):
    if method == EntailmentMethod.DOC:
        x[f"{metric_name}_score"] = scores[idx]
    if method == EntailmentMethod.SENT:
        x[f"{metric_name}_precision"] = precisions[idx]
        x[f"{metric_name}_recall"] = recalls[idx]
        x[f"{metric_name}_f1"] = f1s[idx]
        x[f"{metric_name}_summary_alignment"] = summary_alignments[idx]
        x[f"{metric_name}_summary_entailment"] = summary_entailments[idx]
        x[f"{metric_name}_source_alignment"] = source_alignments[idx]
        x[f"{metric_name}_source_entailment"] = source_entailments[idx]
with open(f"{metric_name}.json", "w", encoding="UTF-8") as outfile:
    json.dump(data, outfile)

# Calculate correlation
table = [["Method", "Pearson", "Spearman"]]
if method == EntailmentMethod.DOC:
    for scores, label in zip([scores], ["_score"]):
        table.append([metric_name + label, pearson(scores, faithfulness_scores), spearman(scores, faithfulness_scores)])

if method == EntailmentMethod.SENT:
    for scores, label in zip([precisions, recalls, f1s], ["_precision", "_recall", "_f1"]):
        table.append([metric_name + label, pearson(scores, faithfulness_scores), spearman(scores, faithfulness_scores)])

# Save correlation as csv file
with open(f"{metric_name}.csv", "w", encoding="UTF-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(table)

print("Finished! :)")
