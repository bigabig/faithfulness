def visualize(data):
    return {
        "source_sentences": data["source_sentences"],
        "summary_sentences": data["summary_sentences"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "summary_source_alignment": data["summary_source_alignment"],
        "source_summary_alignment": data["source_summary_alignment"],
        "summary_source_similarities": data["summary_source_similarities"],
        "source_summary_similarities": data["source_summary_similarities"],
    }
