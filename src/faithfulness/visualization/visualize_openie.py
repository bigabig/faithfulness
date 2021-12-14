def visualize(data):
    # assign ids
    summary_triples = []
    for idx, triple in enumerate(data["summary_triples"]):
        triple["tokens"] = []
        triple["tokens"].extend(triple.pop("S"))
        triple["tokens"].extend(triple.pop("R"))
        triple["tokens"].extend(triple.pop("O"))
        triple["tokens"] = sorted(triple["tokens"], key=lambda t: t["start"], reverse=True)
        triple["id"] = f"summary-triple-{idx}"
        summary_triples.append(triple)

    source_triples = []
    for idx, triple in enumerate(data["source_triples"]):
        triple["tokens"] = []
        triple["tokens"].extend(triple.pop("S"))
        triple["tokens"].extend(triple.pop("R"))
        triple["tokens"].extend(triple.pop("O"))
        triple["tokens"] = sorted(triple["tokens"], key=lambda t: t["start"], reverse=True)
        triple["id"] = f"source-triple-{idx}"
        source_triples.append(triple)

    # resolve alignment
    for summary_id, source_id in enumerate(data[f"summary_source_alignment"]):
        summary_triples[summary_id]["alignment"] = f"source-triple-{source_id}"
        summary_triples[summary_id]["similarity"] = data[f"summary_source_similarities"][summary_id][source_id]

    for source_id, summary_id in enumerate(data[f"source_summary_alignment"]):
        source_triples[source_id]["alignment"] = f"summary-triple-{summary_id}"
        source_triples[source_id]["similarity"] = data[f"source_summary_similarities"][source_id][summary_id]

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
        triples.sort(key=lambda e: e["similarity"] if "similarity" in e else 0.0, reverse=True)

    for triples in grouped_source_triples.values():
        triples.sort(key=lambda e: e["similarity"] if "similarity" in e else 0.0, reverse=True)

    return {
        "summary_sentences": data["new_summary_sentences"],
        "source_sentences": data["new_source_sentences"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "summary_triples": grouped_summary_triples,
        "source_triples": grouped_source_triples,
    }
