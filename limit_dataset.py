import os
import torch
import random
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils import data as udata
from rich import print
from tqdm.rich import tqdm
import typer
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])


def num_limit_and_output_dataset(
    full_dataset_path: str,
    output_path: str,
    gen_num: int,
    train_rate: float = 0.9,
    test_rate: float = 0.1,
    random_seed: int = 2023,
):
    full_dataset = ImageFolder(full_dataset_path, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    indices = []
    for i in range(num_classes):
        class_indices = [
            idx for idx, label in enumerate(full_dataset.targets) if label == i
        ]
        random.shuffle(class_indices)
        class_indices = class_indices[:gen_num]
        indices += class_indices
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(
        full_dataset, sampler=sampler, batch_size=1
    )

    for idx, (img, label) in enumerate(tqdm(dataloader)):
        os.makedirs(f"{output_path}/{label}", exist_ok=True)
        save_image(img, f"{output_path}/{label}/image_{idx:03d}.jpg")

    print("dataset save finished.")


if __name__ == "__main__":
    typer.run(num_limit_and_output_dataset)
