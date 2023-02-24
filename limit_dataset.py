import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils import data as udata
from rich import print
from tqdm.rich import tqdm
import typer


def num_limit_and_output_dataset(
    full_dataset_path: str,
    output_path: str,
    gen_num: int,
    train_rate: float = 0.9,
    test_rate: float = 0.1,
    random_seed: int = 2023,
):
    sampler_idx = [i for i in range(gen_num)]
    full_dataset = ImageFolder(full_dataset_path)
    data_sampler = udata.RandomSampler(
        sampler_idx, generator=torch.Generator().manual_seed(random_seed)
    )
    data_loader = udata.DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        sampler=data_sampler,
    )
    for idx, (img, label) in enumerate(tqdm(data_loader)):
        os.makedirs(f"{output_path}/{label}", exist_ok=True)
        img.save(f"{output_path}/{label}/image_{idx:03d}.jpg")

    print("test dataset save finished.")


if __name__ == "__main__":
    typer.run(num_limit_and_output_dataset)
