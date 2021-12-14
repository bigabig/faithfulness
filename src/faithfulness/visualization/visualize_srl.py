def visualize(data):
    source_phrases = data["source_phrases"]
    summary_phrases = data["summary_phrases"]

    # assign ids
    for srl_label, entities in summary_phrases.items():
        for idx, entity in enumerate(entities):
            entity["id"] = f"summary-{srl_label}-{idx}"

    for srl_label, entities in source_phrases.items():
        for idx, entity in enumerate(entities):
            entity["id"] = f"source-{srl_label}-{idx}"

    # resolve alignment
    for srl_label, alignment in data[f"summary_source_alignment"].items():
        for summary_id, source_id in enumerate(alignment):
            summary_phrases[srl_label][summary_id]["alignment"] = f"source-{srl_label}-{source_id}"
            summary_phrases[srl_label][summary_id]["similarity"] = \
            data[f"summary_source_similarities"][srl_label][summary_id][source_id]

    for srl_label, alignment in data[f"source_summary_alignment"].items():
        for source_id, summary_id in enumerate(alignment):
            source_phrases[srl_label][source_id]["alignment"] = f"summary-{srl_label}-{summary_id}"
            source_phrases[srl_label][source_id]["similarity"] = \
            data[f"source_summary_similarities"][srl_label][source_id][summary_id]

    # group by sentence id
    sum_phr = {}
    for phrases in summary_phrases.values():
        for phrase in phrases:
            sentence = phrase["sentence"]
            sum_phr[sentence] = [*sum_phr.get(sentence, []), phrase]
    sum_phr = dict(sorted(sum_phr.items()))

    src_phr = {}
    for phrases in source_phrases.values():
        for phrase in phrases:
            sentence = phrase["sentence"]
            src_phr[sentence] = [*src_phr.get(sentence, []), phrase]
    src_phr = dict(sorted(src_phr.items()))

    # sort phrases descending by start
    for phrases in src_phr.values():
        phrases.sort(key=lambda e: e["start"], reverse=True)

    for phrases in sum_phr.values():
        phrases.sort(key=lambda e: e["start"], reverse=True)

    return {
        "source_sentences": data["new_source_sentences"],
        "summary_sentences": data["new_summary_sentences"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "summary_phrases": sum_phr,
        "source_phrases": src_phr
    }
