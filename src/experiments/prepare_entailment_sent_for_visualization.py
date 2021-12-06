import json


def main():
    metricname = "entailment_sent"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "entailment_sent",
        "data": [],
    }
    for x in data:
        # delete unnecessary information
        x.pop("source")
        x.pop("summary")

        # rename fields
        x["score"] = x.pop(f"{metricname}_score")
        x["alignment"] = x.pop(f"{metricname}_alignment")
        x["entailment"] = x.pop(f"{metricname}_entailment")

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
