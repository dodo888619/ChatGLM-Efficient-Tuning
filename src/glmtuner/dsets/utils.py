from typing import Dict
from datasets import Dataset


def split_dataset(
    dataset: Dataset, dev_ratio: float, do_train: bool
) -> Dict[str, Dataset]:
    if not do_train:
        return {"eval_dataset": dataset}
    if dev_ratio <= 1e-6:
        return {"train_dataset": dataset}
    dataset = dataset.train_test_split(test_size=dev_ratio)
    return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
