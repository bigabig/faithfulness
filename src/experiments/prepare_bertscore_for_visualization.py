import json
import numpy as np
from transformers import AutoTokenizer


def convert_token_to_word_similarities(tokens, similarities, tokenizer):
    word_id = 0
    start_new = True

    # results
    words = []
    tokenid2wordid = {}

    # word info
    current_word = ""
    current_tokens = []
    current_similarity = -1
    current_similar_to = -1

    def finish_word():
        nonlocal current_word
        nonlocal current_tokens
        nonlocal current_similarity
        nonlocal current_similar_to
        nonlocal word_id

        if len(current_word) > 0:
            words.append([current_word, current_similarity, current_similar_to])
            for tid in current_tokens:
                tokenid2wordid[tid] = word_id

            current_word = ""
            current_tokens = []
            current_similarity = -1
            current_similar_to = -1
            word_id += 1

    for idx, (token, similarity) in enumerate(zip(tokens, similarities)):
        text = tokenizer.decode(token)

        if text.startswith(" "):
            finish_word()
            start_new = True

        if start_new:
            # start new word
            tmp = np.array(similarity)
            current_similarity = tmp.max()
            current_similar_to = tmp.argmax()
            start_new = False

        # append to current word
        current_word += text
        current_tokens.append(idx)

        if text.startswith("."):
            finish_word()
            start_new = True

    finish_word()

    return words, tokenid2wordid


def prepare(summary_tokens, source_tokens, similarities, tokenizer):
    summary_words, summary_tokenid2summary_wordid = convert_token_to_word_similarities(summary_tokens, similarities, tokenizer)
    source_words, source_tokenid2source_wordid = convert_token_to_word_similarities(source_tokens, np.array(similarities).T.tolist(), tokenizer)

    for summary_word in summary_words:
        summary_word[2] = source_tokenid2source_wordid[summary_word[2]]

    for source_word in source_words:
        source_word[2] = summary_tokenid2summary_wordid[source_word[2]]

    return summary_words, source_words


def main():
    # Load input data
    with open("bertscore_sent.json", "r", encoding="UTF-8") as infile:
        data = json.load(infile)
    data = data[:10]

    # Load BERTScore tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=False)

    # Prepare the data
    metricname = "bertscore_sent"
    result = {
        "method": "bertscore",
        "data": [],
    }
    for x in data:
        summary_words, source_words = prepare(x[f"{metricname}_summary_tokens"], x[f"{metricname}_source_tokens"], x[f"{metricname}_similarities"], tokenizer)

        result["data"].append({
            "summary_words": summary_words,
            "source_words": source_words,
            "faithfulness": x["faithfulness"],
            "precision": x[f"{metricname}_precision"],
            "recall": x[f"{metricname}_recall"],
            "f1": x[f"{metricname}_f1"]
        })

    # Write output as json
    with open(f"{metricname}_ui.json", "w", encoding="UTF-8") as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()
