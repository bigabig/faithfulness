from typing import List
from torch.utils.data import DataLoader
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import SummarizationDataset
from faithfulness.utils.utils import calc_prf1, F1Result
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch


class SentCos(SimilarityMetricInterface):

    def __init__(self, modelname: str = "all-mpnet-base-v2", batch_size=2):
        print(f"Loading sentence embedding model {modelname}...")
        self.modelname = modelname
        self.model = SentenceTransformer(modelname)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

    def score(self, summary_text: str, source_text: str) -> F1Result:
        summary_embeddings = self.model.encode([summary_text], convert_to_tensor=True, device=self.device)
        source_embeddings = self.model.encode([source_text], convert_to_tensor=True, device=self.device)
        return self.__calc_score(source_embeddings, summary_embeddings)

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[F1Result]:
        assert len(summaries) == len(sources)

        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=self.batch_size, shuffle=False)

        results: List[F1Result] = []
        for batch in dataloader:
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]

            summaries_embeddings = self.model.encode(batch_summaries, convert_to_tensor=True, device=self.device)
            sources_embeddings = self.model.encode(batch_sources, convert_to_tensor=True, device=self.device)

            # compare summary and source pairwise
            for summary_embeddings, source_embeddings in zip(summaries_embeddings, sources_embeddings):
                results.append(self.__calc_score(source_embeddings, summary_embeddings))

        return results

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]) -> AlignScoreResult:
        if len(summary_texts) == 0 or len(source_texts) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "summary_source_alignment": [],
                "source_summary_alignment": [],
                "summary_source_similarities": [],
                "source_summary_similarities": []
            }

        summary_embeddings = self.model.encode(summary_texts, convert_to_tensor=True, device=self.device, batch_size=self.batch_size)
        source_embeddings = self.model.encode(source_texts, convert_to_tensor=True, device=self.device, batch_size=self.batch_size)

        # compute similarities
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1).tolist()
        source_summary_alignment = similarities.argmax(dim=0).tolist()

        # compute scores
        prf1 = calc_prf1(similarities)

        return {
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1": prf1["f1"],
            "summary_source_alignment": summary_source_alignment,
            "source_summary_alignment": source_summary_alignment,
            "summary_source_similarities": similarities.tolist(),
            "source_summary_similarities": similarities.T.tolist(),
        }

    @staticmethod
    def __calc_score(source_embeddings, summary_embeddings) -> F1Result:
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)
        return {"f1": calc_prf1(similarities)["f1"]}  # f1, precision, recall is the same anyway!
