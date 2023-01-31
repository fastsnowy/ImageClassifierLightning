import typer
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils import data as udata
from rich import print
from tqdm.rich import tqdm


def split_and_output_dataset(
    full_dataset_path: str,
    output_path: str,
    train_rate: float = 0.9,
    test_rate: float = 0.1,
    random_seed: int = 2023,
):
    full_dataset = ImageFolder(full_dataset_path)
    train_dataset, test_dataset = udata.random_split(
        full_dataset,
        [train_rate, test_rate],
        torch.Generator().manual_seed(random_seed),
    )
    print("save test dataset")
    for idx, (img, label) in enumerate(tqdm(test_dataset)):
        os.makedirs(f"{output_path}/test/{label}", exist_ok=True)
        img.save(f"{output_path}/test/{label}/image_{idx:03d}.jpg")

    print("test dataset save finished.")
    print("save train dataset...")
    for idx, (img, label) in enumerate(tqdm(train_dataset)):
        os.makedirs(f"{output_path}/train/{label}", exist_ok=True)
        img.save(f"{output_path}/train/{label}/image_{idx:03d}.jpg")
    print("train dataset save finished.")


if __name__ == "__main__":
    typer.run(split_and_output_dataset)
