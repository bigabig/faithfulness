from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import SummarizationDataset
from faithfulness.utils.utils import calc_prf1, MetricVariant
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch


class SentCos(SimilarityMetricInterface):

    def __init__(self, modelname: str = "all-mpnet-base-v2", variant=MetricVariant.F1, batch_size=2):
        print(f"Loading sentence embedding model {modelname}...")
        self.modelname = modelname
        self.model = SentenceTransformer(modelname)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.variant = variant
        self.batch_size = batch_size

    def get_variant(self) -> MetricVariant:
        return self.variant

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        summary_embeddings = self.model.encode([summary_text], convert_to_tensor=True, device=self.device)
        source_embeddings = self.model.encode([source_text], convert_to_tensor=True, device=self.device)
        return self.__calc_score(source_embeddings, summary_embeddings, additional_output)

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        assert len(summaries) == len(sources)

        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=self.batch_size, shuffle=False)

        results = {}
        for batch in tqdm(dataloader,  desc="Calculating SentCos..."):
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]

            summaries_embeddings = self.model.encode(batch_summaries, convert_to_tensor=True, device=self.device)
            sources_embeddings = self.model.encode(batch_sources, convert_to_tensor=True, device=self.device)

            # compare summary and source pairwise
            for summary_embeddings, source_embeddings in zip(summaries_embeddings, sources_embeddings):
                result = self.__calc_score(source_embeddings, summary_embeddings, additional_output)
                for key, value in result.items():
                    results[key] = [*results.get(key, []), value]

        return results

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        if len(summary_texts) == 0 or len(source_texts) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "alignment": [], "similarities": []}

        summary_embeddings = self.model.encode(summary_texts, convert_to_tensor=True, device=self.device, batch_size=self.batch_size)
        source_embeddings = self.model.encode(source_texts, convert_to_tensor=True, device=self.device, batch_size=self.batch_size)

        # compute similarities
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1).tolist()
        # source_summary_alignment = similarities.argmax(dim=0).tolist()

        # compute scores
        precision, recall, f1 = calc_prf1(similarities)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alignment": summary_source_alignment,
            "similarities": similarities.tolist()
        }

    def __calc_score(self, source_embeddings, summary_embeddings, additional_output: bool):
        similarities = util.pytorch_cos_sim(summary_embeddings, source_embeddings)
        result = calc_prf1(similarities, named=True)
        if additional_output:
            return result
        return {self.variant.value: result[self.variant.value]}
