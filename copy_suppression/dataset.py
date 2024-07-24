import torch as t
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset


class OpenWebTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index) -> Int[Tensor, "max_seq_len"]:
        is_single = isinstance(index, int)
        index = [index] if is_single else index
        items = self.__getitems__(index, unbind=False)
        return items[0] if is_single else items

    def __getitems__(
        self, index, unbind=True
    ) -> Int[Tensor, "batch max_seq_len"]:
        text = self._dataset[index]["text"]
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len - 1,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        bos_tokens = t.empty(
            (
                len(index),
                self.max_seq_len,
            ),
            dtype=t.long,
        )
        bos_tokens[:, 0] = self.tokenizer.eos_token_id
        bos_tokens[:, 1:] = tokens

        return bos_tokens.unbind(dim=0) if unbind else bos_tokens
