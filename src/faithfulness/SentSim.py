import spacy
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface


class SentSim:

    def __init__(self, metric: SimilarityMetricInterface, spacymodel='en_core_web_lg'):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric

    def eval(self, summary_text, source_text):
        # split sentences
        summary_sentences = self.__split_sentences(summary_text)
        source_sentences = self.__split_sentences(source_text)

        return self.metric.align_and_score(summary_sentences, source_sentences)

    def __split_sentences(self, text):
        return [x.text for x in self.nlp(text).sents]

