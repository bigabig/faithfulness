import pathlib
from enum import Enum
from typing import List, Type, Dict
import torch
from allennlp.predictors.predictor import Predictor
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.interfaces.UsesSimilarityMetricInterface import UsesSimilarityMetricInterface
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.types.GroupedAlignScoreResult import GroupedAlignScoreResult
from faithfulness.utils.utils import load_data, save_data, ensure_dir_exists, PRF1Result
from tqdm import tqdm


class SRLPhrase:

    def __init__(self, text: str, start: int, end: int, tag: str, sentence: int, idx: int):
        self.text = text
        self.start = start
        self.end = end
        self.tag = tag
        self.sentence = sentence
        self.idx = idx

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash(('start', self.start,
                     'end', self.end))

    def __str__(self):
        return f"Start: {self.start}, End: {self.end}, Text: {self.text}, Tag: {self.tag}"


class SRLIntermediateResult(GroupedAlignScoreResult):
    summary_phrases:  Dict[str, List[SRLPhrase]]
    source_phrases:  Dict[str, List[SRLPhrase]]


class SRLResult(SRLIntermediateResult):
    summary_sentences: List[str]
    source_sentences: List[str]


class SRLMethod(Enum):
    GROUPS = 1
    NO_GROUPS = 2


class SRL(MetricInterface, UsesSimilarityMetricInterface):

    def __init__(self, save_path: pathlib.Path, metric: Type[SimilarityMetricInterface], metric_args=None, method=SRLMethod.GROUPS, model_path="../models/structured-prediction-srl-bert.2020.12.15.tar.gz", batch_mode=False):
        super(SRL, self).__init__(metric=metric, metric_args=metric_args)
        self.device = 0 if torch.cuda.is_available() else -1

        self.save_path = save_path
        ensure_dir_exists(save_path)

        self.method: SRLMethod = method

        self.model_path = model_path
        self.batch_mode = batch_mode
        self.model = None
        if not self.batch_mode:
            self.__load_srl_model()
            self.load_metric()

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.SENTENCE

    def score(self, summary_sentences: List[str], source_sentences: List[str]) -> SRLResult:
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        summary_phrases, new_summary_sentences = self.__get_phrases(summary_sentences)
        source_phrases, new_source_sentences = self.__get_phrases(source_sentences)
        result = self.__calc_score(summary_phrases, source_phrases)
        return dict(result, summary_sentences=new_summary_sentences, source_sentences=new_source_sentences)

    def score_batch(self, summaries_sentences: List[List[str]], sources_sentences: List[List[str]]) -> List[SRLResult]:
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        sources_phrases: List[Dict[str, List[SRLPhrase]]] = load_data(self.save_path / "sources_phrases.pkl")
        new_sources_sentences: List[List[str]] = load_data(self.save_path / "sources_sentences.pkl")
        if len(sources_phrases) == 0:
            self.__load_srl_model()
            last_input: List[str] = []
            last_phrases: Dict[str, List[SRLPhrase]] = {}
            last_new_sentences: List[str] = []
            for source_sentences in tqdm(sources_sentences, desc="Extracting source phrases..."):

                if len(source_sentences) == len(last_input) and source_sentences[0] == last_input[0]:
                    phrases = last_phrases
                    new_sentences = last_new_sentences
                    print("Skip")
                else:
                    phrases, new_sentences = self.__get_phrases(source_sentences)
                    last_phrases = phrases
                    last_new_sentences = new_sentences
                    last_input = source_sentences

                sources_phrases.append(phrases)
                new_sources_sentences.append(new_sentences)

            save_data(sources_phrases, self.save_path / "sources_phrases.pkl")
            save_data(new_sources_sentences, self.save_path / "sources_sentences.pkl")

        summaries_phrases: List[Dict[str, List[SRLPhrase]]] = load_data(self.save_path / "summaries_phrases.pkl")
        new_summaries_sentences: List[List[str]] = load_data(self.save_path / "summaries_sentences.pkl")
        if len(summaries_phrases) == 0:
            self.__load_srl_model()
            for summary_sentences in tqdm(summaries_sentences, desc="Extracting summary phrases..."):
                phrases, new_sentences = self.__get_phrases(summary_sentences)
                summaries_phrases.append(phrases)
                new_summaries_sentences.append(new_sentences)
            save_data(summaries_phrases, self.save_path / "summaries_phrases.pkl")
            save_data(new_summaries_sentences, self.save_path / "summaries_sentences.pkl")

        results: List[SRLResult] = []
        self.load_metric()
        for summary_phrases, source_phrases, new_sum_sent, new_src_sent in tqdm(zip(summaries_phrases, sources_phrases, new_summaries_sentences, new_sources_sentences), desc="Calculating scores..."):
            result = self.__calc_score(summary_phrases, source_phrases)
            results.append(dict(result, summary_sentences=new_sum_sent, source_sentences=new_src_sent))

        return results

    def __calc_score(self, summary_phrases: Dict[str, List[SRLPhrase]], source_phrases: Dict[str, List[SRLPhrase]]) -> SRLIntermediateResult:
        # use different groups if method = SRLMethod.GROUPS
        # use only one group (which is effectively, no groups at all) if method == SRLMethod.NO_GROUPS
        if self.method == SRLMethod.NO_GROUPS:
            all_summary_phrases = []
            for phrases in summary_phrases.values():
                all_summary_phrases.extend(phrases)
            summary_phrases = {"all": all_summary_phrases}

            all_source_phrases = []
            for phrases in source_phrases.values():
                all_source_phrases.extend(phrases)
            source_phrases = {"all": all_source_phrases}

        # return default result, if no phrases and no sources
        if len(summary_phrases) == 0 or len(source_phrases) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "summary_source_alignment": {},
                "source_summary_alignment": {},
                "summary_source_similarities": {},
                "source_summary_similarities": {},
                "summary_phrases": summary_phrases,
                "source_phrases": source_phrases
            }

        # Find common and missing tags
        common_tags = set.intersection(set(summary_phrases.keys()), set(source_phrases.keys()))
        missing_tags = set(summary_phrases.keys()).difference(set(source_phrases.keys()))

        results: SRLIntermediateResult = {}
        for tag in common_tags:
            result = self.metric.align_and_score([argument.text for argument in summary_phrases[tag]], [argument.text for argument in source_phrases[tag]])
            for key, value in result.items():
                if key in ["precision", "recall", "f1"]:
                    results[key] = [*results.get(key, []), value]
                else:  # key in ["summary_source_alignment", "source_summary_alignment", "summary_source_similarities", "source_summary_similarities"
                    if key not in results:
                        results[key] = {}
                    results[key][tag] = value

        # TODO: find out if this is actually helpful!
        for _ in missing_tags:
            for key in ["precision", "recall", "f1"]:
                results[key] = [*results.get(key, []), 0.0]

        # average scores
        for score_label in ["precision", "recall", "f1"]:
            if score_label in results and len(results[score_label]) > 0:
                results[score_label] = sum(results[score_label]) / len(results[score_label])
            else:
                results[score_label] = 0.0

        results["summary_phrases"] = summary_phrases
        results["source_phrases"] = source_phrases
        return results

    def __get_phrases(self, sentences: List[str]) -> (Dict[str, List[SRLPhrase]], List[str]):
        # Predict frames
        sentences = [{"sentence": sentence} for sentence in sentences]
        predictions = self.model.predict_batch_json(sentences)

        # Parse frames
        all_frames: List[List[SRLPhrase]] = []
        new_sentences: List[str] = []
        for sent_id, pred in enumerate(predictions):
            frames = set()
            words = pred['words']
            for x in pred['verbs']:
                tags = x['tags']
                frame = self.__parse_frame_new(words, tags, sent_id)
                # frame = self.__merge_arguments(frame)
                frames.update(frame)
            new_sentences.append(" ".join(words))
            all_frames.append(SRL.find_sentence_representation(list(frames)))

        # order phrases by tag
        phrases: Dict[str, List[SRLPhrase]] = {}
        for sentence_phrases in all_frames:
            for phrase in sentence_phrases:
                phrases[phrase.tag] = [*phrases.get(phrase.tag, []), phrase]

        # filters and groups phrases by tag
        phrases = self.__filter_and_group_phrases_by_tag(phrases)

        return phrases, new_sentences

    @staticmethod
    def __parse_frame_new(words: List[str], tags: List[str], sentence_id: int):
        chunk = []
        results = []
        current_tag = ""
        idx = 0

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    text = " ".join(chunk)
                    start = idx
                    end = idx + len(text) + 1
                    idx = end
                    results.append(SRLPhrase(text, start, end, current_tag, sentence_id, len(results)))
                    chunk = []

                if tag.startswith("B-"):
                    current_tag = tag[2:]
                    if current_tag.startswith("C-"):
                        current_tag = tag[2:]
                    chunk.append(token)
                elif tag == "O":
                    idx += len(token) + 1

        if chunk:
            text = " ".join(chunk)
            start = idx
            end = idx + len(text)
            results.append(SRLPhrase(text, start, end, current_tag, sentence_id, len(results)))

        return results

    @staticmethod
    def __best_schedule(lst: List[SRLPhrase]) -> List[SRLPhrase]:
        if len(lst) == 0:
            return lst

        lst.sort(key=lambda x: x.start, reverse=True)

        last_char = max(map(lambda x: x.end, lst))
        data: List[List[SRLPhrase]] = [[] for _ in range(last_char + 1)]

        for span in lst:
            current_schedule = data[span.start]
            new_schedule: List[SRLPhrase] = [span] + data[span.end]
            # the longer schedule is the better one
            # however, if both schedules include the same number of tasks
            # the schedule that takes more time is more time is better
            if len(new_schedule) == len(current_schedule):
                new_coverage = sum(map(lambda x: x.end - x.start, new_schedule))
                current_coverage = sum(map(lambda x: x.end - x.start, current_schedule))
                best_schedule = new_schedule if new_coverage > current_coverage else current_schedule
            elif len(new_schedule) > len(current_schedule):
                best_schedule = new_schedule
            else:
                best_schedule = current_schedule

            # set data
            for i in range(span.start + 1):
                data[i] = best_schedule

        return data[0]

    @staticmethod
    def find_sentence_representation(frames: List[SRLPhrase]) -> List[SRLPhrase]:
        best_frames = SRL.__best_schedule(frames)

        result: List[SRLPhrase] = []
        # merge frames of same tag that occur exactly next to each other
        if len(best_frames) > 0:
            prev = best_frames[0]
            for i in range(len(best_frames) - 1):
                current = best_frames[i + 1]
                if prev.tag == current.tag and prev.end == current.start:
                    prev = SRLPhrase(prev.text + " " + current.text, prev.start, current.end, current.tag, current.sentence, 0)
                else:
                    result.append(prev)
                    prev = current
            result.append(prev)

        return result

    @staticmethod
    def __filter_and_group_phrases_by_tag(phrases: Dict[str, List[SRLPhrase]]) -> Dict[str, List[SRLPhrase]]:
        attributes: List[SRLPhrase] = []
        negations: List[SRLPhrase] = []
        directions: List[SRLPhrase] = []
        reasons: List[SRLPhrase] = []
        locations: List[SRLPhrase] = []
        subjects: List[SRLPhrase] = []
        verbs: List[SRLPhrase] = []
        objects: List[SRLPhrase] = []
        hows: List[SRLPhrase] = []
        whens: List[SRLPhrase] = []

        for tag in phrases.keys():
            if tag in ['ARG2', 'ARG3', 'ARG4', 'ARGM-PRD']:
                attributes.extend(phrases[tag])
            elif tag in ['ARGM-NEG']:
                negations.extend(phrases[tag])
            elif tag in ['ARGM-DIR']:
                directions.extend(phrases[tag])
            elif tag in ['ARGM-CAU', 'ARGM-PRP', 'ARGM-PNC', 'ARGM-GOL']:
                reasons.extend(phrases[tag])
            elif tag in ['ARGM-LOC']:
                locations.extend(phrases[tag])
            elif tag in ['ARG0', 'ARGM-COM', 'ARGA']:
                subjects.extend(phrases[tag])
            elif tag in ['V']:
                verbs.extend(phrases[tag])
            elif tag in ['ARG1']:
                objects.extend(phrases[tag])
            elif tag in ['ARGM-MNR', 'ARGM-EXT', 'ARGM-ADV', 'ARGM-ADJ']:
                hows.extend(phrases[tag])
            elif tag in ['ARGM-TMP']:
                whens.extend(phrases[tag])

        result: Dict[str, List[SRLPhrase]] = {}
        if len(attributes) > 0:
            result['attributes'] = attributes
        if len(negations) > 0:
            result['negations'] = negations
        if len(directions) > 0:
            result['directions'] = directions
        if len(reasons) > 0:
            result['reasons'] = reasons
        if len(locations) > 0:
            result['locations'] = locations
        if len(subjects) > 0:
            result['subjects'] = subjects
        if len(verbs) > 0:
            result['verbs'] = verbs
        if len(objects) > 0:
            result['objects'] = objects
        if len(hows) > 0:
            result['hows'] = hows
        if len(whens) > 0:
            result['whens'] = whens
        return result

    def __load_srl_model(self):
        if self.model is None:
            print(f"Loading semantic role labeling model ...")
            self.model = Predictor.from_path(self.model_path, cuda_device=self.device)

    def __load_metric(self):
        if self.metric is None:
            self.metric = self.metric_type(*self.metric_args)
