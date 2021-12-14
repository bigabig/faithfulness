def visualize(data):
    return {
        "source_sentences": data["source_sentences"],
        "summary_sentences": data["summary_sentences"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "source_alignment": data["source_alignment"],
        "summary_alignment": data["summary_alignment"],
        "source_entailment": data["source_entailment"],
        "summary_entailment": data["summary_entailment"],
    }
