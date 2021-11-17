# from QGQA import QGQA
from faithfulness.SRL import SRL
from faithfulness.similarity.SentCos import SentCos
from faithfulness.utils.utils import MetricVariant


def main():
    summary = "Tim works for the University of Hamburg. He lives in Lübeck."
    source = "Since 2021, Tim Fischer works at the University of Hamburg, but he still lives in Lübeck."
    summaries = ["Tim is a nice human.", "He lives in Lübeck.", "He works in Hamburg"]
    sources = ["Tim is a very kind person.", "Tim Fischer lives in Lübeck but works in Hamburg.", "Tim Fischer works at the University of Hamburg"]

    # import stanza
    # stanza.install_corenlp()

    # ie = OpenIE(metric=F1())
    # precision, recall, f1, alignment, similarities = ie.eval(summary, source)
    # print("LOL")

    # BERTScore for alignment + similarity Example
    # bs = BERTScore()
    # precision, recall, f1, summary_source_alignment, source_summary_alignment, similarities = bs.align_and_score(summaries, sources)
    # print(f"The overall faithfulness is {f1 * 100:.2f}%. The summary faithfulness is {precision * 100:.2f}%, the source faithfulness {recall * 100:.2f}%.")
    # print("Alignment: (summary -> source, similarity)")
    # for summary_id, source_id in enumerate(summary_source_alignment):
    #     print(f"{summaries[summary_id]} -> {sources[source_id]} | {similarities[summary_id][source_id] * 100:.2f}%")

    # BERTScore batch eval example
    # bs = BERTScore()
    # result = bs.eval_batch(summaries, sources)  # [[precision, recall, f1], ...]
    # for idx, scores in enumerate(result):
    #     print(f"Summary {idx} has a faithfulness of {scores[2]}.")

    # Entailment Example - Method = Sent

    # entailment = Entailment(method=EntailmentMethod.SENT)
    # faithfulness, entailed_by, scores = entailment.eval(summary, source)
    # print(f"The summary faithfulness is {faithfulness}")
    # for i, (ent, score) in enumerate(zip(entailed_by, scores)):
    #     print(f"Summary sentence {i} is entailed by source sentence {ent} with a probability of {score}.")

    # Entailment Example - Method = Doc
    # entailment = Entailment(method=EntailmentMethod.DOC)
    # faithfulness = entailment.eval(summary, source)
    # print(f"The summary faithfulness is {faithfulness}")

    # QGQA Example
    # qgqa = QGQA(metric=F1())
    # faithfulness, info = qgqa.eval(summary, source)
    # print(f"The summary faithfulness is {faithfulness}")

    # SRL Example

    scos = SentCos()
    print(scos.score_batch(summaries, sources))

    # srl = SRL(metric=SentCos())
    # faithfulness = srl.eval(summary, source)
    # print(f"The summary faithfulness is {faithfulness}")

    # NER Example
    # ner = NER()
    # faithfulness = ner.eval(summary, source)
    # print(f"The summary faithfulness is {faithfulness}")


if __name__ == '__main__':
    main()
