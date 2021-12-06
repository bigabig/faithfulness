import json


def main():
    metricname = "qgqa_f1"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "qgqa",
        "data": [],
    }
    for x in data:
        # delete unnecessary information
        x.pop("source_sentences")
        x.pop("summary_sentences")

        # rename fields
        x["score"] = x.pop(f"{metricname}_score")
        x["questions"] = x.pop(f"{metricname}_questions")

        # filter questions: we only want unique summary_answer, source_answer pairs!
        questions = {}
        for q in x['questions']:
            pair = ((q['answer'], q['text_answer']) )
            if pair not in questions.keys():
                q['id'] = len(questions)
                q.pop("score")
                q.pop("expected_answer")
                questions[pair] = q
        x["questions"] = list(questions.values())

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
