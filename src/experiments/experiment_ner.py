import json
import csv
from faithfulness.BERTScore import BERTScore, BERTScoreMethod

# Load input data
from faithfulness.NER import NER
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.utils.correlation import pearson, spearman

with open("prepared_xsum.json", "r", encoding="UTF-8") as infile:
    data = json.load(infile)
summaries, sources, faithfulness_scores = zip(*[(x["summary"], x["source"], x["faithfulness"]) for x in data])

# Load metric
metric_name = "ner_em"
similarity_metric = ExactMatch()
metric = NER(metric=similarity_metric)

# Calculate faithfulness
summaries_entities, sources_entities, \
    precisions, recalls, f1s, \
    summary_source_alignments, summary_source_similarities, \
    source_summary_alignments, source_summary_similarities = metric.score_batch(summaries, sources, True).values()

# Save results as json file
for (idx, x) in enumerate(data):
    x[f"{metric_name}_precision"] = precisions[idx]
    x[f"{metric_name}_recall"] = recalls[idx]
    x[f"{metric_name}_f1"] = f1s[idx]
    x[f"{metric_name}_summary_entities"] = summaries_entities[idx]
    x[f"{metric_name}_source_entities"] = sources_entities[idx]
    x[f"{metric_name}_summary_source_similarities"] = summary_source_similarities[idx]
    x[f"{metric_name}_source_summary_similarities"] = source_summary_similarities[idx]
    x[f"{metric_name}_summary_source_alignment"] = summary_source_alignments[idx]
    x[f"{metric_name}_source_summary_alignment"] = source_summary_alignments[idx]
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
