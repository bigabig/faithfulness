from typing import Type, Optional
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface


class UsesSimilarityMetricInterface:

    def __init__(self, metric: Type[SimilarityMetricInterface], metric_args=None):
        self.metric_type = metric
        if metric_args is None:
            self.metric_args = {}
        else:
            self.metric_args = metric_args

        self.metric: Optional[SimilarityMetricInterface] = None

    def load_metric(self):
        if self.metric is None:
            self.metric = self.metric_type(*self.metric_args)
