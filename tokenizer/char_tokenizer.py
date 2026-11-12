
import json
from pathlib import Path


class CharTokenizer:

    def __init__(self, text=None, vocab=None):

        self.unk_token = "<UNK>"

        if vocab is not None:
            self.vocab = vocab
        elif text is not None:
            self.vocab = sorted(set(text))
            self.vocab.append(self.unk_token)
        else:
            raise ValueError("Provide either text or vocab")

        self.char_to_id = {
            char: idx for idx, char in enumerate(self.vocab)
        }

        self.id_to_char = {
            idx: char for idx, char in enumerate(self.vocab)
        }

    def encode(self, text):

        unk_id = self.char_to_id[self.unk_token]

        return [
            self.char_to_id.get(char, unk_id)
            for char in text
        ]

    def decode(self, ids):

        return ''.join(
            self.id_to_char[int(idx)]
            for idx in ids
        )

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                self.vocab,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):

        with open(path, "r", encoding="utf-8") as file:
            vocab = json.load(file)

        return cls(vocab=vocab)
