from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SummarizationDataset(Dataset):
    def __init__(self, summaries, sources):
        self.summaries = summaries
        self.sources = sources
        assert len(summaries) == len(sources)

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return {
            "summaries": self.summaries[idx],
            "sources": self.sources[idx]
        }


class QADataset(Dataset):
    def __init__(self, questions, contexts):
        self.question = questions
        self.context = contexts
        assert len(questions) == len(contexts)

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        return {
            "question": self.question[idx],
            "context": self.context[idx]
        }


class QGDataset(Dataset):
    def __init__(self, sentences, answers):
        self.sentences = sentences
        self.answers = answers
        assert len(self.sentences) == len(self.answers)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "answer": self.answers[idx]
        }
