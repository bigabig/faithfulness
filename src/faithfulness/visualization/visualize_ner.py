def visualize(data):
    source_entities = data["source_entities"]
    summary_entities = data["summary_entities"]

    # assign ids
    for entity_type, entities in summary_entities.items():
        for idx, entity in enumerate(entities):
            entity["id"] = f"summary-{entity_type}-{idx}"

    for entity_type, entities in source_entities.items():
        for idx, entity in enumerate(entities):
            entity["id"] = f"source-{entity_type}-{idx}"

    # resolve alignment
    for entity_type, alignment in data["summary_source_alignment"].items():
        for summary_id, source_id in enumerate(alignment):
            summary_entities[entity_type][summary_id]["alignment"] = f"source-{entity_type}-{source_id}"
            summary_entities[entity_type][summary_id]["similarity"] = data["summary_source_similarities"][entity_type][summary_id][source_id]

    for entity_type, alignment in data["source_summary_alignment"].items():
        for source_id, summary_id in enumerate(alignment):
            source_entities[entity_type][source_id]["alignment"] = f"summary-{entity_type}-{summary_id}"
            source_entities[entity_type][source_id]["similarity"] = data["source_summary_similarities"][entity_type][source_id][summary_id]

    # flatten
    sum_ents = []
    for entities in summary_entities.values():
        sum_ents.extend(entities)

    src_ents = []
    for entities in source_entities.values():
        src_ents.extend(entities)

    # sort entities descending by start
    sum_ents.sort(key=lambda e: e["start"], reverse=True)
    src_ents.sort(key=lambda e: e["start"], reverse=True)

    return {
        "source": data["source"],
        "summary": data["summary"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "summary_entities": sum_ents,
        "source_entities": src_ents,
    }
