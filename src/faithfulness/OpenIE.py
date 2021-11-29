import pickle
from typing import List, Type
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from stanza.server import CoreNLPClient
import numpy as np
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.similarity.SentCos import SentCos
from faithfulness.utils.utils import MetricVariant, load_data, save_data
from tqdm import tqdm


class OpenIE(MetricInterface):

    def __init__(self, metric: Type[SimilarityMetricInterface], variant=MetricVariant.F1, modelname: str = "paraphrase-mpnet-base-v2", batch_mode=False, save_path="", metric_args=None):
        self.metric_type = metric
        if metric_args is None:
            self.metric_args = {}
        else:
            self.metric_args = metric_args

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_mode = batch_mode
        self.variant = variant
        self.save_path = save_path
        self.modelname = modelname

        self.metric = None
        self.client = None
        self.model = None
        if not batch_mode:
            # Connect to CoreNLP
            self.__load_core_nlp()

            # Load sentence embedding model
            self.__load_sentence_embedder()

            # Load metric
            self.__load_metric()

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        summary_triples = self.__get_triples(summary_text)  # [{'S': ..., 'R': ..., 'O': ...}, ...]
        source_triples = self.__get_triples(source_text)

        summary_triples = self.__filter_triples(summary_triples)
        source_triples = self.__filter_triples(source_triples)

        # could be used e.g. for Relation Matching Rate
        # summary_triples = [triple[1] for triple in summary_triples]
        # source_triples = [triple[1] for triple in source_triples]

        return self.__calc_score(summary_triples, source_triples, additional_output)

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        summaries_triples = load_data(self.save_path + "_summaries_triples.pkl")
        if len(summaries_triples) == 0:
            self.__load_core_nlp()
            for summary in tqdm(summaries, desc="Extracting summary triples..."):
                summaries_triples.append(self.__get_triples(summary))
            save_data(summaries_triples, self.save_path + "_summaries_triples.pkl")

        sources_triples = load_data(self.save_path + "_sources_triples.pkl")
        start_idx = len(sources_triples)
        if len(sources_triples) != len(sources):
            if start_idx > 0:
                print(f"Continuing extracting source triples from id {start_idx}")
            self.__load_core_nlp()
            for source in tqdm(sources[start_idx:], desc="Extracting source triples..."):
                sources_triples.append(self.__get_triples(source))
                save_data(sources_triples, self.save_path + "_sources_triples.pkl")

        summaries_triples_filtered = load_data(self.save_path + "_summary_triples_filtered.pkl")
        if len(summaries_triples_filtered) == 0:
            self.__load_sentence_embedder()
            for summary_triples in tqdm(summaries_triples, desc="Filtering summary triples..."):
                summaries_triples_filtered.append(self.__filter_triples(summary_triples))
            save_data(summaries_triples_filtered, self.save_path + "_summary_triples_filtered.pkl")

        sources_triples_filtered = load_data(self.save_path + "_sources_triples_filtered.pkl")
        start_idx2 = len(sources_triples_filtered)
        if len(sources_triples_filtered) != len(sources):
            if start_idx2 > 0:
                print(f"Continuing filtering source triples from id {start_idx2}")
            self.__load_sentence_embedder()
            for source_triples in tqdm(sources_triples[start_idx2:], desc="Filtering source triples..."):
                sources_triples_filtered.append(self.__filter_triples(source_triples))
                save_data(sources_triples_filtered, self.save_path + "_sources_triples_filtered.pkl")

        results = {}
        self.__load_metric()
        for summary_triples, source_triples in tqdm(zip(summaries_triples_filtered, sources_triples_filtered), desc="Calculating scores..."):
            result = self.__calc_score(summary_triples, source_triples, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]

        return results

    def __get_triples(self, text):
        if len(text) > 5000:
            text = text[:5000]

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

        clustering = DBSCAN(min_samples=1).fit_predict(similarities.cpu())

        cluster = {}
        for cluster_id, sentence, triple in zip(clustering, sentences, triples):
            if cluster_id not in cluster.keys():
                cluster[cluster_id] = []
            cluster[cluster_id].append((sentence, triple))

        # we only use the longest triple of each cluster (as longer = more information)
        filtered_triples = [value[np.array([len(tup[0]) for tup in value]).argmax()] for key, value in cluster.items()]
        return filtered_triples

    def __calc_score(self, summary_triples: List[str], source_triples: List[str] , additional_output: bool):
        summary_triple_sentences = [triple[0] for triple in summary_triples]
        source_triple_sentences = [triple[0] for triple in source_triples]
        result = self.metric.align_and_score(summary_triple_sentences, source_triple_sentences)
        if additional_output:
            result["summary_triples"] = summary_triples
            result["source_triples"] = source_triples
            return result
        else:
            return {self.variant.value: result[self.variant.value]}

    def __load_metric(self):
        if self.metric is None:
            self.metric = self.metric_type(*self.metric_args)

    def __load_core_nlp(self):
        if self.client is None:
            self.client = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse', 'ner', 'coref', 'natlog', 'openie'],
                timeout=30000,
                properties={'openie.resolve_coref': True},
                memory='8G')

    def __load_sentence_embedder(self):
        if self.model is None:
            print(f"Loading sentence embedding model {self.modelname}...")
            self.model = SentenceTransformer(self.modelname)
            self.model.to(self.device)

            # # do not load sentence embedding model twice, if metric is SentCos to save resources
            # if isinstance(self.metric, SentCos) and self.modelname == self.metric.modelname:
            #     print(f"Skipping loading sentence embedding model {self.modelname}!")
            #     self.model = self.metric.model
