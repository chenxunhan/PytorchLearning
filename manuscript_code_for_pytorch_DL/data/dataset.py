import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_my_dataset(dataset_name="FashionMNIST", need_download=True):
    if dataset_name == "FashionMNIST":
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="./download_path",
            train=True,
            download=need_download,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="./download_path",
            train=False,
            download=need_download,
            transform=ToTensor(),
        )

        num_examples = {"training_data": len(training_data), "testset": len(test_data)}
        return training_data, test_data, num_examples

    elif dataset_name == "CIFAR10":

        my_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Download training data from open datasets.
        training_data = datasets.CIFAR10(root="./dataset", train=True, download=True, transform=my_transform)
        # Download test data from open datasets.
        test_data = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=my_transform)

        num_examples = {"training_data": len(training_data), "testset": len(test_data)}
        return training_data, test_data, num_examples


    # TODO
    else:
        dataset = CustomImageDataset()
        return dataset

