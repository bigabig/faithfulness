from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", use_fast=False)


def convert_token_to_word_similarities(tokens, similarities):
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


def prepare(summary_tokens, source_tokens, similarities):
    summary_words, summary_tokenid2summary_wordid = convert_token_to_word_similarities(summary_tokens, similarities)
    source_words, source_tokenid2source_wordid = convert_token_to_word_similarities(source_tokens, np.array(similarities).T.tolist())

    for summary_word in summary_words:
        summary_word[2] = source_tokenid2source_wordid[summary_word[2]]

    for source_word in source_words:
        source_word[2] = summary_tokenid2summary_wordid[source_word[2]]

    return summary_words, source_words


def visualize(data):
    summary_words, source_words = prepare(data["summary_tokens"],
                                          data["source_tokens"],
                                          data["similarities"])

    return {
        "summary_words": summary_words,
        "source_words": source_words,
        "faithfulness": data["faithfulness"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"]
    }
