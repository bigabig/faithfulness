import json


def main():
    metricname = "openie_sentcos_test"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "openie",
        "data": [],
    }
    for x in data:

        # assign ids
        summary_triples = []
        for idx, triple in enumerate(x[f"{metricname}_summary_triples"]):
            triple["tokens"] = []
            triple["tokens"].extend(triple.pop("S"))
            triple["tokens"].extend(triple.pop("R"))
            triple["tokens"].extend(triple.pop("O"))
            triple["tokens"] = sorted(triple["tokens"], key=lambda x: x["start"], reverse=True)
            triple["id"] = f"summary-triple-{idx}"
            summary_triples.append(triple)

        source_triples = []
        for idx, triple in enumerate(x[f"{metricname}_source_triples"]):
            triple["tokens"] = []
            triple["tokens"].extend(triple.pop("S"))
            triple["tokens"].extend(triple.pop("R"))
            triple["tokens"].extend(triple.pop("O"))
            triple["tokens"] = sorted(triple["tokens"], key=lambda x: x["start"], reverse=True)
            triple["id"] = f"source-triple-{idx}"
            source_triples.append(triple)

        # resolve alignment
        for summary_id, source_id in enumerate(x[f"{metricname}_alignment"]):
            summary_triples[summary_id]["alignment"] = f"source-triple-{source_id}"
            summary_triples[summary_id]["similarity"] = x[f"{metricname}_similarity"][summary_id][source_id]

        # group by sentence id
        grouped_summary_triples = {}
        for triple in summary_triples:
            sentence = triple["sentence"]
            grouped_summary_triples[sentence] = [*grouped_summary_triples.get(sentence, []), triple]
        grouped_summary_triples = dict(sorted(grouped_summary_triples.items()))

        grouped_source_triples = {}
        for triple in source_triples:
            sentence = triple["sentence"]
            grouped_source_triples[sentence] = [*grouped_source_triples.get(sentence, []), triple]
        grouped_source_triples = dict(sorted(grouped_source_triples.items()))

        # sort phrases descending by similarity
        for triples in grouped_summary_triples.values():
            triples.sort(key=lambda e: e["similarity"], reverse=True)

        # for triples in grouped_source_triples.values():
        #     triples.sort(key=lambda e: e["similarity"], reverse=True)

        x["summary_triples"] = grouped_summary_triples
        x["source_triples"] = grouped_source_triples

        # delete unnecessary information
        # x.pop("source")
        # x.pop("summary")
        x.pop(f"{metricname}_similarity")
        x.pop(f"{metricname}_alignment")
        x.pop(f"{metricname}_summary_triples")
        x.pop(f"{metricname}_source_triples")

        # rename fields
        x["summary_sentences"] = x.pop(f"{metricname}_summary_sentences")
        x["source_sentences"] = x.pop(f"{metricname}_source_sentences")
        x["precision"] = x.pop(f"{metricname}_precision")
        x["recall"] = x.pop(f"{metricname}_recall")
        x["f1"] = x.pop(f"{metricname}_f1")

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
