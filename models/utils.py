import torch
import torch.nn as nn
import torch.utils


def get_device(allow_tf32=True, precision="medium"):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            # Allow TensorFloat32 on matmul and convolutions
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.set_float32_matmul_precision(precision)
        return device


class DummyTokenizer(nn.Module):
    def __init__(self, tokenizer_fn):
        super().__init__()
        self.tokenizer_fn = tokenizer_fn

    def forward(self, input):
        output = [self.tokenizer_fn(sentence) for sentence in input]
        return output


class Truncate(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input):
        output = [sentence[: self.max_seq_len] for sentence in input]
        return output


class AddToken(nn.Module):
    def __init__(self, token_id: str, begin: bool = False):
        super().__init__()
        self.token_id = token_id
        self.begin = begin

    def forward(self, input):
        output = []
        if self.begin:
            for ids in input:
                output.append([self.token_id] + ids)
        else:
            for ids in input:
                output.append(ids + [self.token_id])

        return output


class VocabTransform(nn.Module):
    def __init__(self, word2idx: dict, unk_tok: int):
        super().__init__()
        self.word2idx = word2idx
        self.unk_tok = unk_tok

    def forward(self, input):
        output = []
        for sentence in input:
            output.append([self.word2idx.get(word, self.unk_tok) for word in sentence])
        return output


class ToTensor(nn.Module):
    def __init__(self, padding_value: int, dtype: torch.dtype):
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input):
        if self.padding_value is None:
            output = torch.tensor(input, dtype=self.dtype)
            return output
        else:
            output = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=self.dtype) for ids in input],
                batch_first=True,
                padding_value=float(self.padding_value),
            )
            return output


class PadTransform(nn.Module):
    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x):
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x
