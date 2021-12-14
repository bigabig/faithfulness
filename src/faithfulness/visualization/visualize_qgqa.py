def visualize(data):
    # filter questions: we only want unique summary_answer, source_answer pairs!
    questions = {}
    for q in data['questions']:
        pair = ((q['answer'], q['text_answer']))
        if pair not in questions.keys():
            q['id'] = len(questions)
            q.pop("score")
            q.pop("expected_answer")
            questions[pair] = q

    return {
        "summary": data["summary"],
        "source": data["source"],
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "questions": list(questions.values())
    }
