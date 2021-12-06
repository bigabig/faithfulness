from typing import List, Union
import sympy

class Phrase:
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


def permutation(lst: List[Phrase]):
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
        permutations = permutation(rem_lst)
        if len(permutations) == 0:
            permu.append([m])
        else:
            for p in permutations:
                permu.append([m] + p)
    return permu


def find_groups(frames: List[Phrase]):
    intervals = [sympy.Interval(x.start, x.end - 1) for x in frames]
    u = sympy.Union(*intervals)
    groups = [u] if isinstance(u, sympy.Interval) else list(u.args)

    res = {}
    for frame in frames:
        for group_num, l in enumerate(groups):
            if l.contains(frame.start) and l.contains(frame.end - 1):
                res[group_num] = [*res.get(group_num, []), frame]

    return res


def scale(value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> float:
    if min_value == max_value:
        return 1.0 * value
    return 1.0 * (value - min_value) / (1.0 * (max_value - min_value))


def find_sentence_representation(frames: List[Phrase]):
    groups = find_groups(frames)
    print(f"Frame: {len(frames)}, Groups: {len(groups)}")

    group_ids = sorted(list(groups.keys()))

    best_representation = []
    for group_num in group_ids:
        fs = groups[group_num]

        permutations = permutation(fs)

        if len(permutations) == 0:
            return []

        # find best compromise between number of frames and text length
        stats = [(len(x), sum([len(frame.text) for frame in x]), x) for x in permutations]

        max_num_frames = max(stats, key=lambda x: x[0])[0]
        min_num_frames = min(stats, key=lambda x: x[0])[0]

        max_text_length = max(stats, key=lambda x: x[1])[1]
        min_text_length = min(stats, key=lambda x: x[1])[1]

        stats = [(scale(x[0], min_num_frames, max_num_frames) + scale(x[1], min_text_length, max_text_length),
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
                prev = Phrase(prev.text + " " + current.text, prev.start, current.end, current.tag, current.sentence)
            else:
                result.append(prev)
                prev = current
        result.append(prev)

    return result
