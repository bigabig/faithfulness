import random
from enum import Enum
from typing import List
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.utils.utils import PRF1Result


class BaselineMethod(Enum):
    ZERO = 1,
    ONE = 2,
    RANDOM = 3


class Baseline(MetricInterface):

    def __init__(self, method=BaselineMethod.RANDOM):
        self.method = method

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.DOCUMENT

    def score(self, summary: str, source: str) -> PRF1Result:
        if self.method == BaselineMethod.ONE:
            value = 1.0
        elif self.method == BaselineMethod.ZERO:
            value = 0.0
        else:
            value = random.uniform(0.0, 1.0)

        # has to return PRF1 to be compatible with Experimentator
        return {
            "precision": value,
            "recall": value,
            "f1": value
        }

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[PRF1Result]:
        return [self.score(summary, source) for summary, source in zip(summaries, sources)]
