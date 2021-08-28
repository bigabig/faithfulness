import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from stanza.server import CoreNLPClient
import numpy as np
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface


class OpenIE:

    def __init__(self, metric: SimilarityMetricInterface, modelname: str = "paraphrase-mpnet-base-v2"):
        self.metric = metric

        # Connect to CoreNLP
        self.client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse', 'ner', 'coref', 'natlog', 'openie'],
            timeout=30000,
            properties={'openie.resolve_coref': True},
            memory='8G')

        # Load sentence embedding model
        # todo: wird aktuell 2x geladen wenn metric = SentCos :/
        print(f"Loading sentence embedding model {modelname}...")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(modelname)
        self.model.to(self.device)

    def eval(self, summary_text, source_text):
        summary_triples = self.__get_triples(summary_text)  # [{'S': ..., 'R': ..., 'O': ...}, ...]
        source_triples = self.__get_triples(source_text)

        summary_triples = self.__filter_triples(summary_triples)
        source_triples = self.__filter_triples(source_triples)

        summary_triple_sentences = [triple[0] for triple in summary_triples]
        source_triple_sentences = [triple[0] for triple in source_triples]

        # could be used e.g. for Relation Matching Rate
        # summary_triples = [triple[1] for triple in summary_triples]
        # source_triples = [triple[1] for triple in source_triples]

        return self.metric.align_and_score(summary_triple_sentences, source_triple_sentences)

    def __get_triples(self, text):
        # annotate the data
        ann = self.client.annotate(text)

        # extract triples
        triples = []
        for sentence in ann.sentence:
            for triple in sentence.openieTriple:
                triples.append({'S': triple.subject, 'R': triple.relation, 'O': triple.object})

        return triples

    def __filter_triples(self, triples):
        if len(triples) == 0:
            return triples

        sentences = [f"{triple['S']} {triple['R']} {triple['O']}." for triple in triples]
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
        similarities = (1 - util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)) * 2
        # similarities = 1 - similarities
        # similarities = similarities * 2

        clustering = DBSCAN(min_samples=1).fit_predict(similarities.cpu())

        cluster = {}
        for cluster_id, sentence, triple in zip(clustering, sentences, triples):
            if cluster_id not in cluster.keys():
                cluster[cluster_id] = []
            cluster[cluster_id].append((sentence, triple))

        # we only use the longest triple of each cluster (as longer = more information)
        filtered_triples = [value[np.array([len(tup[0]) for tup in value]).argmax()] for key, value in cluster.items()]
        return filtered_triples
