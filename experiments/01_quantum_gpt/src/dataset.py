import os
import hashlib
import torch
from datasets import load_dataset as load_hf_dataset
from src.tokenizer import CharTokenizer, BiCharTokenizer, HFTokenizerWrapper


class InputDataset:
    def __init__(self, config, file_path_or_repo, dictionary_path=None, seed=1337):
        self.config = config
        self.device = config.device
        self.block_size = config.block_size
        self.reset_generators(seed)

        self.tokenizer = self._setup_tokenizer(
            config, dictionary_path, file_path_or_repo
        )
        raw_text = self._load_raw_data(file_path_or_repo)
        self.source_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        print(f"Tokenizing dataset (Vocab size: {self.tokenizer.vocab_size})...")
        full_data = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)

        # 90/10 train/val split
        n = int(0.9 * len(full_data))
        self.train_data = full_data[:n]
        self.val_data = full_data[n:]
        self.n_tokens = len(full_data)

    def reset_generators(self, seed):
        """Reset split-specific batch sampling without touching model RNG state."""
        self.generators = {
            "train": torch.Generator().manual_seed(seed),
            "val": torch.Generator().manual_seed(seed + 1),
        }

    def _setup_tokenizer(self, config, dict_path, data_sample):
        t_type = config.tokenizer_class

        if t_type.startswith("hf-") or t_type in ["gpt2", "roberta-base"]:
            return HFTokenizerWrapper(t_type.replace("hf-", ""))

        tokenizers_map = {
            "CharTokenizer": CharTokenizer,
            "BiCharTokenizer": BiCharTokenizer,
        }
        cls = tokenizers_map.get(t_type, CharTokenizer)
        tokenizer = cls()

        if dict_path and os.path.exists(dict_path):
            import json

            with open(dict_path, "r") as f:
                tokenizer.from_dict(json.load(f))
        else:
            tokenizer = cls(self._load_raw_data(data_sample))

        return tokenizer

    def _load_raw_data(self, path):
        """Load text from a local file or HF Hub dataset."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif os.path.exists(f"data/{path}"):
            with open(f"data/{path}", "r", encoding="utf-8") as f:
                return f.read()

        print(f"File '{path}' not found. Attempting to load from HF Hub...")
        ds = load_hf_dataset(path, split="train")
        column_names = ds.column_names

        target_column = None
        for candidate in ["text", "Text", "content", "body", "document"]:
            if candidate in column_names:
                target_column = candidate
                break

        if not target_column:
            for col in column_names:
                if isinstance(ds[0][col], str):
                    target_column = col
                    break

        if not target_column:
            raise ValueError(
                f"Could not find a text column in dataset '{path}'. "
                f"Available columns: {column_names}"
            )

        print(f"Found text in column: '{target_column}'")
        return "\n".join(ds[target_column])

    def get_batch(self, split, batch_size):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.block_size,
            (batch_size,),
            generator=self.generators[split],
        )
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
