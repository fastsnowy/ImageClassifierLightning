import os

import torch
import typer
from rich import print
from torch.utils import data as udata
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm.rich import tqdm


def create_class_balanced_datasets(
    dataset_dir, train_ratio, batch_size, num_workers, seed
):
    """
    dataset_dir: データセットのパス
    train_ratio: トレーニングデータセットの比率
    batch_size: バッチサイズ
    num_workers: データローダーの並列数
    seed: seed
    """
    # 画像データを読み込むための変換
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # データセットを読み込む
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)

    # クラスごとに分割する
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx
    class_datasets = {class_: [] for class_ in classes}
    for data, target in dataset:
        class_datasets[classes[target]].append((data, target))

    # トレーニングデータセットとテストデータセットを作成する
    train_datasets = []
    test_datasets = []
    for class_, data in class_datasets.items():
        num_data = len(data)
        num_train = int(num_data * train_ratio)
        num_test = num_data - num_train
        train_data, test_data = udata.random_split(
            data, [num_train, num_test], generator=torch.Generator().manual_seed(seed)
        )
        train_datasets.append(train_data)
        test_datasets.append(test_data)

    # すべてのクラスのトレーニングデータセットとテストデータセットを組み合わせる
    train_dataset = udata.ConcatDataset(train_datasets)
    test_dataset = udata.ConcatDataset(test_datasets)

    # データローダーを作成する
    train_loader = udata.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = udata.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, class_to_idx


def split(
    full_dataset_path: str,
    output_path: str,
    train_rate: float = 0.9,
    random_seed: int = 2023,
):
    train_loader, test_loader, _ = create_class_balanced_datasets(
        full_dataset_path,
        train_rate,
        1,
        8,
        random_seed,
    )

    print("save train dataset...")
    for idx, (img, label) in enumerate(tqdm(train_loader)):
        os.makedirs(f"{output_path}/train/{label.item()}", exist_ok=True)
        save_image(img, f"{output_path}/train/{label.item()}/image_{idx:03d}.jpg")

    print("save test dataset...")
    for idx, (img, label) in enumerate(tqdm(test_loader)):
        os.makedirs(f"{output_path}/test/{label.item()}", exist_ok=True)
        save_image(img, f"{output_path}/test/{label.item()}/image_{idx:03d}.jpg")


if __name__ == "__main__":
    typer.run(split)
