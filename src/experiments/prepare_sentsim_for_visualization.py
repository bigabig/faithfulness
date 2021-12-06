import json


def main():
    metricname = "sentsim_sentcos_all-mpnet-base-v2"

    # Load input data
    with open(f"{metricname}.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Prepare the data
    result = {
        "method": "sentsim",
        "data": [],
    }
    for x in data:
        # delete unnecessary information
        x.pop("source")
        x.pop("summary")

        # rename fields
        x["precision"] = x.pop(f"{metricname}_precision")
        x["recall"] = x.pop(f"{metricname}_recall")
        x["f1"] = x.pop(f"{metricname}_f1")
        x["alignment"] = x.pop(f"{metricname}_alignment")
        x["similarity"] = x.pop(f"{metricname}_similarity")

        result["data"].append(x)

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
