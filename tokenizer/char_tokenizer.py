
class CharTokenizer:
    def __init__(self, text):
        self.unk_token = "<UNK>"

        # Build vocabulary
        self.vocab = sorted(set(text))
        self.vocab.append(self.unk_token)

        # Character-to-ID mapping
        self.char_to_id = {
            char: idx for idx, char in enumerate(self.vocab)
        }

        # ID-to-character mapping
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
            self.id_to_char[idx]
            for idx in ids
        )
