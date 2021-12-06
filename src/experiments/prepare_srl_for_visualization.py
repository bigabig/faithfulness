import json


def main():
    metricname = "srl_em_test"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "srl",
        "data": [],
    }
    for x in data:

        # assign ids
        for phrase_type, phrases in x[f"{metricname}_summary_phrases"].items():
            for idx, phrase in enumerate(phrases):
                phrase["id"] = f"summary-{phrase_type}-{idx}"

        for phrase_type, phrases in x[f"{metricname}_source_phrases"].items():
            for idx, phrase in enumerate(phrases):
                phrase["id"] = f"source-{phrase_type}-{idx}"

        # resolve alignment
        for phrase_type, alignment in x[f"{metricname}_alignment"].items():
            for summary_id, source_id in enumerate(alignment):

                summary_phrase = x[f"{metricname}_summary_phrases"][phrase_type][summary_id]
                summary_phrase["alignment"] = f"source-{phrase_type}-{source_id}"
                summary_phrase["similarity"] = x[f"{metricname}_similarities"][phrase_type][summary_id][source_id]

        # group by sentence id
        summary_phrases = {}
        for phrases in x[f"{metricname}_summary_phrases"].values():
            for phrase in phrases:
                sentence = phrase["sentence"]
                summary_phrases[sentence] = [*summary_phrases.get(sentence, []), phrase]
        summary_phrases = dict(sorted(summary_phrases.items()))

        source_phrases = {}
        for phrases in x[f"{metricname}_source_phrases"].values():
            for phrase in phrases:
                sentence = phrase["sentence"]
                source_phrases[sentence] = [*source_phrases.get(sentence, []), phrase]
        source_phrases = dict(sorted(source_phrases.items()))

        # sort phrases descending by start
        for phrases in source_phrases.values():
            phrases.sort(key=lambda e: e["start"], reverse=True)

        for phrases in summary_phrases.values():
            phrases.sort(key=lambda e: e["start"], reverse=True)

        x["summary_phrases"] = summary_phrases
        x["source_phrases"] = source_phrases

        # delete unnecessary information
        x.pop("source")
        x.pop("summary")
        x.pop(f"{metricname}_similarities")
        x.pop(f"{metricname}_alignment")
        x.pop(f"{metricname}_summary_phrases")
        x.pop(f"{metricname}_source_phrases")

        # rename fields
        x["summary_sentences"] = x.pop(f"{metricname}_new_summary_sentences")
        x["source_sentences"] = x.pop(f"{metricname}_new_source_sentences")
        x["precision"] = x.pop(f"{metricname}_precision")
        x["recall"] = x.pop(f"{metricname}_recall")
        x["f1"] = x.pop(f"{metricname}_f1")

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
