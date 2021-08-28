from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
