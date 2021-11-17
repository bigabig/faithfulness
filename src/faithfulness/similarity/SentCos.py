from typing import List
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import calc_prf1
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch


class SentCos(SimilarityMetricInterface):

    def __init__(self, modelname: str = "paraphrase-mpnet-base-v2"):
        print(f"Loading sentence embedding model {modelname}...")
        self.model = SentenceTransformer(modelname)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def score(self, summary_text: str, source_text: str):
        summary_embeddings = self.model.encode([summary_text], convert_to_tensor=True, device=self.device)
        source_embeddings = self.model.encode([source_text], convert_to_tensor=True, device=self.device)
        return self.__calc_sent_sim(source_embeddings, summary_embeddings)  # returns f1 (but prf1 all are the same actually)

    def score_batch(self, summaries: List[str], sources: List[str]):
        assert len(summaries) == len(sources)

        summaries_embeddings = self.model.encode(summaries, convert_to_tensor=True, device=self.device)
        sources_embeddings = self.model.encode(sources, convert_to_tensor=True, device=self.device)

        # compare summary and source pairwise
        return [self.__calc_sent_sim(summary_embeddings, source_embeddings) for summary_embeddings, source_embeddings in zip(summaries_embeddings, sources_embeddings)]

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        summary_embeddings = self.model.encode(summary_texts, convert_to_tensor=True, device=self.device)
        source_embeddings = self.model.encode(source_texts, convert_to_tensor=True, device=self.device)

        # compute similarities
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1).tolist()
        # source_summary_alignment = similarities.argmax(dim=0).tolist()

        # compute scores
        precision, recall, f1 = calc_prf1(similarities)

        return precision, recall, f1, summary_source_alignment, similarities.tolist()

    @staticmethod
    def __calc_sent_sim(summary_embeddings, source_embeddings):
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)
        return similarities.max(dim=1).values.mean()  # this is precision, but recall and f1 are the same anyway
