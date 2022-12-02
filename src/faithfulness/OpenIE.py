import sys
import pathlib
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from stanza.server import CoreNLPClient
import numpy as np
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.interfaces.UsesSimilarityMetricInterface import UsesSimilarityMetricInterface
from faithfulness.similarity.SentCos import SentCos
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.utils.representation_utils import Phrase
from faithfulness.utils.utils import load_data, save_data, ensure_dir_exists
from tqdm import tqdm
from typing import List, Type, Dict, Tuple
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class Triple(TypedDict):
    S: List[Phrase]
    R: List[Phrase]
    O: List[Phrase]
    sentence: int
    text: str


class AlignScoreResultWithTriples(AlignScoreResult):
    summary_triples: List[Triple]
    source_triples: List[Triple]


class OpenIEResult(AlignScoreResultWithTriples):
    summary_sentences: List[str]
    source_sentences: List[str]


class OpenIE(MetricInterface, UsesSimilarityMetricInterface):

    def __init__(self, metric: Type[SimilarityMetricInterface], save_path: pathlib.Path, modelname: str = "paraphrase-mpnet-base-v2", batch_mode=False, metric_args=None):
        super(OpenIE, self).__init__(metric, metric_args)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_mode = batch_mode
        self.modelname = modelname

        self.save_path = save_path
        ensure_dir_exists(save_path)

        self.client = None
        self.model = None
        if not batch_mode:
            # Connect to CoreNLP
            self.__load_core_nlp()

            # Load sentence embedding model
            self.__load_sentence_embedder()

            # Load metric
            self.load_metric()

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.DOCUMENT

    def score(self, summary_text: str, source_text: str):
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        summary_triples, summary_sentences = self.__get_triples(summary_text)  # [{'S': ..., 'R': ..., 'O': ...}, ...]
        source_triples, source_sentences = self.__get_triples(source_text)

        summary_triples = self.__filter_triples_new(summary_triples)
        source_triples = self.__filter_triples_new(source_triples)

        # could be used e.g. for Relation Matching Rate
        # summary_triples = [triple[1] for triple in summary_triples]
        # source_triples = [triple[1] for triple in source_triples]

        result = self.__calc_score(summary_triples, source_triples)
        return {
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "summary_source_alignment": result["summary_source_alignment"],
                "source_summary_alignment": result["source_summary_alignment"],
                "summary_source_similarities": result["summary_source_similarities"],
                "source_summary_similarities": result["source_summary_similarities"],
                "source_triples": result["source_triples"],
                "summary_triples": result["summary_triples"],
                "summary_sentences": summary_sentences,
                "source_sentences": source_sentences
            }

    def score_batch(self, summaries: List[str], sources: List[str]):
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        sources_triples: List[List[Triple]] = load_data(self.save_path / "sources_triples.pkl")
        new_sources_sentences: List[List[str]] = load_data(self.save_path / "sources_sentences.pkl")
        start_idx = len(sources_triples)
        if len(sources_triples) != len(sources):
            if start_idx > 0:
                print(f"Continuing extracting source triples from id {start_idx}")
            self.__load_core_nlp()
            for source in tqdm(sources[start_idx:], desc="Extracting source triples..."):
                triples, sentences = self.__get_triples(source)
                sources_triples.append(triples)
                new_sources_sentences.append(sentences)
                save_data(sources_triples, self.save_path / "sources_triples.pkl")
                save_data(new_sources_sentences, self.save_path / "sources_sentences.pkl")

        summaries_triples: List[List[Triple]] = load_data(self.save_path / "summaries_triples.pkl")
        new_summaries_sentences: List[List[str]] = load_data(self.save_path / "summaries_sentences.pkl")
        if len(summaries_triples) == 0:
            self.__load_core_nlp()
            for summary in tqdm(summaries, desc="Extracting summary triples..."):
                triples, sentences = self.__get_triples(summary)
                summaries_triples.append(triples)
                new_summaries_sentences.append(sentences)
            save_data(summaries_triples, self.save_path / "summaries_triples.pkl")
            save_data(new_summaries_sentences, self.save_path / "summaries_sentences.pkl")

        summaries_triples_filtered: List[List[Triple]] = load_data(self.save_path / "summary_triples_filtered.pkl")
        if len(summaries_triples_filtered) == 0:
            self.__load_sentence_embedder()
            for summary_triples in tqdm(summaries_triples, desc="Filtering summary triples..."):
                summaries_triples_filtered.append(self.__filter_triples_new(summary_triples))
            save_data(summaries_triples_filtered, self.save_path / "summary_triples_filtered.pkl")

        sources_triples_filtered: List[List[Triple]] = load_data(self.save_path / "sources_triples_filtered.pkl")
        start_idx2 = len(sources_triples_filtered)
        if len(sources_triples_filtered) != len(sources):
            if start_idx2 > 0:
                print(f"Continuing filtering source triples from id {start_idx2}")
            self.__load_sentence_embedder()
            for source_triples in tqdm(sources_triples[start_idx2:], desc="Filtering source triples..."):
                sources_triples_filtered.append(self.__filter_triples_new(source_triples))
                save_data(sources_triples_filtered, self.save_path / "sources_triples_filtered.pkl")

        results: List[OpenIEResult] = []
        self.load_metric()
        for summary_triples, source_triples, summary_sentences, source_sentences in tqdm(zip(summaries_triples_filtered, sources_triples_filtered, new_summaries_sentences, new_sources_sentences), desc="Calculating scores..."):
            result = self.__calc_score(summary_triples, source_triples)
            results.append({
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "summary_source_alignment": result["summary_source_alignment"],
                "source_summary_alignment": result["source_summary_alignment"],
                "summary_source_similarities": result["summary_source_similarities"],
                "source_summary_similarities": result["source_summary_similarities"],
                "source_triples": result["source_triples"],
                "summary_triples": result["summary_triples"],
                "summary_sentences": summary_sentences,
                "source_sentences": source_sentences
            })

        return results

    def __get_triples(self, text) -> (List[Triple], List[str]):
        if len(text) > 5000:
            text = text[:5000]

        # annotate the data
        ann = self.client.annotate(text)

        # extract triples
        triples: List[Triple] = []
        sentences: List[str] = []
        for sentence_id, sentence in enumerate(ann.sentence):

            sentence_start = sentence.token[0].beginChar
            sentence_end = sentence.token[-1].endChar
            sentence_text = text[sentence_start:sentence_end]
            sentences.append(sentence_text)

            for triple in sentence.openieTriple:
                if triple.subjectTokens and triple.relationTokens and triple.objectTokens:

                    # make sure that tokens are all from the current sentence
                    # we do not want triples that span multiple sentences!
                    sentence_ids = [x.sentenceIndex for x in triple.subjectTokens]
                    sentence_ids.extend([x.sentenceIndex for x in triple.relationTokens])
                    sentence_ids.extend([x.sentenceIndex for x in triple.objectTokens])
                    sentence_ids = list(set(sentence_ids))

                    if len(sentence_ids) == 1 and sentence_ids[0] == sentence_id:
                        subject_tokens = [Phrase(ann.sentence[x.sentenceIndex].token[x.tokenIndex].originalText,
                                                 ann.sentence[x.sentenceIndex].token[
                                                     x.tokenIndex].beginChar - sentence_start,
                                                 ann.sentence[x.sentenceIndex].token[
                                                     x.tokenIndex].endChar - sentence_start,
                                                 "subject",
                                                 sentence_id
                                                 ) for x in triple.subjectTokens]

                        relationTokens = [Phrase(ann.sentence[x.sentenceIndex].token[x.tokenIndex].originalText,
                                                 ann.sentence[x.sentenceIndex].token[
                                                     x.tokenIndex].beginChar - sentence_start,
                                                 ann.sentence[x.sentenceIndex].token[
                                                     x.tokenIndex].endChar - sentence_start,
                                                 "relation",
                                                 sentence_id
                                                 ) for x in triple.relationTokens]

                        objectTokens = [Phrase(ann.sentence[x.sentenceIndex].token[x.tokenIndex].originalText,
                                               ann.sentence[x.sentenceIndex].token[
                                                   x.tokenIndex].beginChar - sentence_start,
                                               ann.sentence[x.sentenceIndex].token[
                                                   x.tokenIndex].endChar - sentence_start,
                                               "object",
                                               sentence_id
                                               ) for x in triple.objectTokens]

                        triples.append(
                            {
                                'S': subject_tokens,
                                'R': relationTokens,
                                'O': objectTokens,
                                'sentence': sentence_id,
                                'text': f"{triple.subject} {triple.relation} {triple.object}."
                            }
                        )

        return triples, sentences

    def __filter_triples_new(self, triples: List[Triple]) -> List[Triple]:
        if len(triples) == 0:
            return triples

        # order triples by sentence
        ordered_triples: Dict[int, List[Triple]] = {}
        for triple in triples:
            sentence_id = triple["sentence"]
            ordered_triples[sentence_id] = [*ordered_triples.get(sentence_id, []), triple]

        # find best triple representations for every sentence (by clustering)
        result: List[Triple] = []
        for sentence_id, ord_tri in ordered_triples.items():
            sentences = [triple["text"] for triple in ord_tri]
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            similarities = (1 - util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)) * 2

            clustering = DBSCAN(min_samples=1).fit_predict(similarities.cpu())

            cluster: Dict[int, List[Tuple[str, Triple]]] = {}
            for cluster_id, sentence, triple in zip(clustering, sentences, ord_tri):
                if cluster_id not in cluster.keys():
                    cluster[cluster_id] = []
                cluster[cluster_id].append((sentence, triple))

            # we only use the longest triple of each cluster (as longer = more information)
            filtered_triples: List[Triple] = [value[np.array([len(tup[0]) for tup in value]).argmax()][1] for key, value
                                              in cluster.items()]
            result.extend(filtered_triples)

        return result

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

    def __calc_score(self, summary_triples: List[Triple], source_triples: List[Triple]) -> AlignScoreResultWithTriples:
        if len(summary_triples) == 0 or len(source_triples) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "summary_source_alignment": [],
                "source_summary_alignment": [],
                "summary_source_similarities": [],
                "source_summary_similarities": [],
                "source_triples": source_triples,
                "summary_triples": summary_triples
            }

        summary_triple_sentences = [triple["text"] for triple in summary_triples]
        source_triple_sentences = [triple["text"] for triple in source_triples]
        result = self.metric.align_and_score(summary_triple_sentences, source_triple_sentences)
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "summary_source_alignment": result["summary_source_alignment"],
            "source_summary_alignment": result["source_summary_alignment"],
            "summary_source_similarities": result["summary_source_similarities"],
            "source_summary_similarities": result["source_summary_similarities"],
            "source_triples": source_triples,
            "summary_triples": summary_triples
        }

    def __load_core_nlp(self):
        if self.client is None:
            self.client = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse', 'ner', 'coref', 'natlog', 'openie'],
                timeout=30000,
                properties={'openie.resolve_coref': True},
                memory='8G')

    def __load_sentence_embedder(self):
        if self.model is None:
            if isinstance(self.metric, SentCos) and self.modelname == self.metric.modelname:
                print(f"Skipping loading sentence embedding model {self.modelname}!")
            else:
                print(f"Loading sentence embedding model {self.modelname}...")
                self.model = SentenceTransformer(self.modelname)
                self.model.to(self.device)
