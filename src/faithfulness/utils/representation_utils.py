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
