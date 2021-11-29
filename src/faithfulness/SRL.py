from typing import List, Type
import torch
from allennlp.predictors.predictor import Predictor
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import load_data, save_data, MetricVariant
from tqdm import tqdm


class SRL(MetricInterface):

    def __init__(self, metric: Type[SimilarityMetricInterface], metric_args=None, model_path="models/structured-prediction-srl-bert.2020.12.15.tar.gz", variant=MetricVariant.F1, batch_mode=False, save_path=""):
        self.metric_type = metric
        self.metric_args = metric_args if metric_args is not None else {}
        self.device = 0 if torch.cuda.is_available() else -1
        self.save_path = save_path
        self.model_path = model_path
        self.variant = variant
        self.batch_mode = batch_mode

        self.metric = None
        self.model = None
        if not self.batch_mode:
            self.__load_srl_model()
            self.__load_metric()

    def score(self, summary_sentences: List[str], source_sentences: List[str], additional_output: bool):
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        summary_phrases = self.__get_phrases(summary_sentences)
        source_phrases = self.__get_phrases(source_sentences)
        return self.__calc_score(summary_phrases, source_phrases, additional_output)

    def score_batch(self, summaries: List[List[str]], sources: List[List[str]], additional_output: bool):
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        summaries_phrases = load_data(self.save_path + "_summaries_phrases.pkl")
        if len(summaries_phrases) == 0:
            self.__load_srl_model()
            for summary_sentences in tqdm(summaries, desc="Extracting summary phrases..."):
                summaries_phrases.append(self.__get_phrases(summary_sentences))
            save_data(summaries_phrases, self.save_path + "_summaries_phrases.pkl")

        sources_phrases = load_data(self.save_path + "_sources_phrases.pkl")
        if len(sources_phrases) == 0:
            self.__load_srl_model()
            for sources_sentences in tqdm(sources, desc="Extracting source phrases..."):
                sources_phrases.append(self.__get_phrases(sources_sentences))
            save_data(sources_phrases, self.save_path + "_sources_phrases.pkl")

        results = {}
        self.__load_metric()
        for summary_phrases, source_phrases in tqdm(zip(summaries_phrases, sources_phrases), desc="Calculating scores..."):
            result = self.__calc_score(summary_phrases, source_phrases, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]

        return results

    def __calc_score(self, summary_phrases: dict, source_phrases: dict, additional_output: bool):
        # Find common and missing tags
        common_tags = set.intersection(set(summary_phrases.keys()), set(source_phrases.keys()))
        missing_tags = set(summary_phrases.keys()).difference(set(source_phrases))

        results = {}
        for tag in common_tags:
            result = self.metric.align_and_score([argument['phrase'] for argument in summary_phrases[tag]], [argument['phrase'] for argument in source_phrases[tag]])
            result["alignment"] = (tag, result["alignment"])
            result["similarities"] = (tag, result["similarities"])
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]

        for _ in missing_tags:
            result = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]

        # average scores
        if len(results) > 0:
            for score_label in ["precision", "recall", "f1"]:
                results[score_label] = sum(results[score_label]) / len(results[score_label])
        else:
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "alignment": [],
                "similarities": [],
            }

        if additional_output:
            results["summary_phrases"] = summary_phrases
            results["source_phrases"] = source_phrases
            return results
        else:
            return {self.variant.value: results[self.variant.value]}

    def __get_phrases(self, sentences: List[str]):
        # Predict frames
        sentences = [{"sentence": sentence} for sentence in sentences]
        predictions = self.model.predict_batch_json(sentences)

        # Parse frames
        all_frames = []
        for pred in predictions:
            words = pred['words']
            for x in pred['verbs']:
                tags = x['tags']
                frame = self.__parse_frame(words, tags)
                frame = self.__merge_arguments(frame)
                all_frames.append(frame)

        # filter frames: we are only interested in frames that have more than one argument
        all_frames = [frame for frame in all_frames if len(frame) > 1]

        # converts phrases to a dict indexed by tag
        phrases = self.__order_phrases_by_tag(all_frames)

        # filters and groups phrases by tag
        phrases = self.__filter_and_group_phrases_by_tag(phrases)

        return phrases

    @staticmethod
    def __parse_frame(words, tags):
        assert len(words) == len(tags)

        frame = []
        current = None
        for idx, (word, tag) in enumerate(zip(words, tags)):
            last_word = idx == len(words) - 1
            splitted = tag.split('-')

            if len(splitted) > 1:
                category = "-".join(splitted[1:])
            else:
                category = splitted[0]

            if current is not None and current['tag'] == category:
                current['words'].append(word)
            elif current is not None and current['tag'] != category:
                current['phrase'] = " ".join(current['words'])
                if not last_word:
                    current['phrase'] = current['phrase'] + " "

                frame.append(current)
                current = {
                    'tag': category,
                    'words': [word]
                }
            else:  # current is None
                current = {
                    'tag': category,
                    'words': [word]
                }

        current['phrase'] = " ".join(current['words'])
        frame.append(current)

        return frame

    @staticmethod
    def __multi_delete(list_, idxs):
        indexes = sorted(idxs, reverse=True)
        for index in indexes:
            del list_[index]
        return list_

    @staticmethod
    def __merge_arguments(frame):
        delete_list = []
        for idx, argument in enumerate(frame):
            if argument['tag'].startswith("C"):
                splitted = argument['tag'].split('-')
                tag = "-".join(splitted[1:])

                merge_with = None
                for y in frame:
                    if y['tag'] == tag:
                        merge_with = y
                        break

                if merge_with is None:
                    argument['tag'] = tag
                else:
                    merge_with['words'].extend(argument['words'])
                    merge_with['phrase'] = " ".join(merge_with['words'])
                    merge_with['phrase'] = merge_with['phrase'] + " "
                    delete_list.append(idx)

        return SRL.__multi_delete(frame, delete_list)

    @staticmethod
    def __order_phrases_by_tag(frames):
        result = {}
        for f in frames:
            for frame in f:
                if frame['tag'] not in result.keys():
                    result[frame['tag']] = []
                result[frame['tag']].append(frame)
        return result

    @staticmethod
    def __filter_and_group_phrases_by_tag(phrases):
        attributes = []
        negations = []
        directions = []
        reasons = []
        locations = []
        subjects = []
        verbs = []
        objects = []
        hows = []
        whens = []

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

        result = {}
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
