from typing import List, Type
import torch
from allennlp.predictors.predictor import Predictor
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import load_data, save_data, MetricVariant
from tqdm import tqdm
from sympy import Interval, Union


class SRLPhrase:

    def __init__(self, text: str, start: int, end: int, tag: str, sentence: int):
        self.text = text
        self.start = start
        self.end = end
        self.tag = tag
        self.sentence = sentence

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash(('start', self.start,
                     'end', self.end))

    def __str__(self):
        return f"Start: {self.start}, End: {self.end}, Text: {self.text}, Tag: {self.tag}"


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

    def score_batch(self, summaries_sentences: List[List[str]], sources_sentences: List[List[str]], additional_output: bool):
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        sources_phrases = load_data(self.save_path + "_sources_phrases.pkl")
        new_sources_sentences = load_data(self.save_path + "_sources_sentences.pkl")
        if len(sources_phrases) == 0:
            self.__load_srl_model()
            for source_sentences in tqdm(sources_sentences, desc="Extracting source phrases..."):
                phrases, new_sentences = self.__get_phrases(source_sentences)
                sources_phrases.append(phrases)
                new_sources_sentences.append(new_sentences)
            save_data(sources_phrases, self.save_path + "_sources_phrases.pkl")
            save_data(new_sources_sentences, self.save_path + "_sources_sentences.pkl")

        summaries_phrases = load_data(self.save_path + "_summaries_phrases.pkl")
        new_summaries_sentences = load_data(self.save_path + "_summaries_sentences.pkl")
        if len(summaries_phrases) == 0:
            self.__load_srl_model()
            for summary_sentences in tqdm(summaries_sentences, desc="Extracting summary phrases..."):
                phrases, new_sentences = self.__get_phrases(summary_sentences)
                summaries_phrases.append(phrases)
                new_summaries_sentences.append(new_sentences)
            save_data(summaries_phrases, self.save_path + "_summaries_phrases.pkl")
            save_data(new_summaries_sentences, self.save_path + "_summaries_sentences.pkl")

        results = {}
        self.__load_metric()
        for summary_phrases, source_phrases in tqdm(zip(summaries_phrases, sources_phrases), desc="Calculating scores..."):
            result = self.__calc_score(summary_phrases, source_phrases, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]

        if additional_output:
            results["summary_sentences"] = new_summaries_sentences
            results["source_sentences"] = new_sources_sentences
        return results

    def __calc_score(self, summary_phrases: dict, source_phrases: dict, additional_output: bool):
        # Find common and missing tags
        common_tags = set.intersection(set(summary_phrases.keys()), set(source_phrases.keys()))
        missing_tags = set(summary_phrases.keys()).difference(set(source_phrases))

        results = {
            "precision": [],
            "recall": [],
            "f1": [],
            "alignment": {},
            "similarities": {},
        }
        for tag in common_tags:

            result = self.metric.align_and_score([argument.text for argument in summary_phrases[tag]], [argument.text for argument in source_phrases[tag]])
            for key, value in result.items():
                if key == "similarities" or key == "alignment":
                    results[key][tag] = result[key]
                else:
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
        for score_label in ["precision", "recall", "f1"]:
            if len(results[score_label]) > 0:
                results[score_label] = sum(results[score_label]) / len(results[score_label])
            else:
                results[score_label] = 0.0

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
        new_sentences = []
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
        phrases = {}
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
                    results.append(SRLPhrase(text, start, end, current_tag, sentence_id))
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
            results.append(SRLPhrase(text, start, end, current_tag, sentence_id))

        return results

    @staticmethod
    def permutation(lst: List[SRLPhrase]):

        # If lst is empty then there are no permutations
        if len(lst) == 0:
            return []

        # If there is only one element in lst then, only
        # one permutation is possible
        if len(lst) == 1:
            return [lst]

        # Find the permutations for lst if there are
        # more than 1 characters
        permu = []  # empty list that will store current permutation

        # Iterate the input(lst) and calculate the permutation
        for i in range(len(lst)):
            m = lst[i]

            # Extract lst[i] or m from the list. rem_lst is remaining list
            rem_lst = lst[:i] + lst[i + 1:]
            rem_lst = [x for x in rem_lst if x.start >= m.end]

            # Generating all permutations where m is first element
            permutations = SRL.permutation(rem_lst)
            if len(permutations) == 0:
                permu.append([m])
            else:
                for p in permutations:
                    permu.append([m] + p)
        return permu

    @staticmethod
    def find_groups(frames):
        intervals = [Interval(x.start, x.end - 1) for x in frames]
        u = Union(*intervals)
        groups = [u] if isinstance(u, Interval) else list(u.args)

        res = {}
        for frame in frames:
            for group_num, l in enumerate(groups):
                if l.contains(frame.start) and l.contains(frame.end - 1):
                    res[group_num] = [*res.get(group_num, []), frame]

        return res

    @staticmethod
    def scale(value, min, max):
        if min == max:
            return value
        return (value - min) / (max - min)

    @staticmethod
    def find_sentence_representation(frames):
        groups = SRL.find_groups(frames)
        print(f"Frame: {len(frames)}, Groups: {len(groups)}")

        group_ids = sorted(list(groups.keys()))

        best_representation = []
        for group_num in group_ids:
            fs = groups[group_num]

            permutations = SRL.permutation(fs)

            if len(permutations) == 0:
                return []

            # find best compromise between number of frames and text length
            stats = [(len(x), sum([len(frame.text) for frame in x]), x) for x in permutations]

            max_num_frames = max(stats, key=lambda x: x[0])[0]
            min_num_frames = min(stats, key=lambda x: x[0])[0]

            max_text_length = max(stats, key=lambda x: x[1])[1]
            min_text_length = min(stats, key=lambda x: x[1])[1]

            stats = [(SRL.scale(x[0], min_num_frames, max_num_frames) + SRL.scale(x[1], min_text_length, max_text_length),
                      x[0], x[1], x[2]) for x in stats]

            best = max(stats, key=lambda x: x[0])[3]
            best_representation.extend(best)

        result = []
        # merge frames of same tag that occur exactly next to each other
        if len(best_representation) > 0:
            prev = best_representation[0]
            for i in range(len(best_representation) - 1):
                current = best_representation[i + 1]
                if prev.tag == current.tag and prev.end == current.start:
                    prev = SRLPhrase(prev.text + " " + current.text, prev.start, current.end, current.tag, current.sentence)
                else:
                    result.append(prev)
                    prev = current
            result.append(prev)

        return result

    @staticmethod
    def convert_to_valid_schedule(schedule: List[SRLPhrase]):
        result = []
        for item in schedule:
            if len(result) == 0 or item.start >= result[-1].end:
                result.append(item)
        return result

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
