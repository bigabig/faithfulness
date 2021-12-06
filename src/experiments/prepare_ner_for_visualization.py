import json


def main():
    metricname = "ner_em2"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "ner",
        "data": [],
    }
    for x in data:
        # resolve alignment
        for entity_type, entities in x[f"{metricname}_summary_entities"].items():
            for entity, similarity, alignment in zip(entities, x[f"{metricname}_summary_source_similarities"][entity_type], x[f"{metricname}_summary_source_alignment"][entity_type]):
                entity["alignment"] = f"source-{entity_type}-{alignment}"
                entity["similarity"] = similarity[alignment]

        for entity_type, entities in x[f"{metricname}_source_entities"].items():
            for entity, similarity, alignment in zip(entities, x[f"{metricname}_source_summary_similarities"][entity_type], x[f"{metricname}_source_summary_alignment"][entity_type]):
                entity["alignment"] = f"summary-{entity_type}-{alignment}"
                entity["similarity"] = similarity[alignment]

        # flatten
        summary_entities = []
        for entities in x[f"{metricname}_summary_entities"].values():
            summary_entities.extend(entities)

        source_entities = []
        for entities in x[f"{metricname}_source_entities"].values():
            source_entities.extend(entities)

        # sort entities descending by start
        summary_entities.sort(key=lambda e: e["start"], reverse=True)
        source_entities.sort(key=lambda e: e["start"], reverse=True)

        x["summary_entities"] = summary_entities
        x["source_entities"] = source_entities

        # delete unnecessary information
        x.pop("source_sentences")
        x.pop("summary_sentences")
        x.pop(f"{metricname}_summary_entities")
        x.pop(f"{metricname}_source_entities")
        x.pop(f"{metricname}_summary_source_alignment")
        x.pop(f"{metricname}_source_summary_alignment")
        x.pop(f"{metricname}_summary_source_similarities")
        x.pop(f"{metricname}_source_summary_similarities")

        # rename fields
        x["precision"] = x.pop(f"{metricname}_precision")
        x["recall"] = x.pop(f"{metricname}_recall")
        x["f1"] = x.pop(f"{metricname}_f1")

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
