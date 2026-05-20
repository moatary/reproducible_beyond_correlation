
"""
Dataset loading helpers shared across all sub-projects.

Datasets
--------
MNIST      : Standard 28x28 grayscale digit images (10 classes).
             Auto-downloaded via torchvision.
Gender Face: 2,500 face images (2 classes: male / female).
             Preprocessed to 28x28 with landmark triangulation,
             histogram equalisation, and rotation correction.
             Place the pickled file at data/imagestargets.pkl.

Splits
------
All splits use an 85 / 15 train-test ratio with a fixed random seed
(see utils/seed.py) to ensure identical splits across runs.
"""
import pickle
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = pathlib.Path('data')


class FlatImageDataset(Dataset):
    """Wraps a numpy array of flattened images with integer labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_mnist(train_size: int = 60_000, test_size: int = 10_000):
    """Return (train_dataset, test_dataset) for MNIST, flattened to 784-dim."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    # Flatten 28x28 → 784
    tr_imgs = train_ds.data.numpy()[:train_size].reshape(train_size, -1).astype(np.float32) / 255.0
    tr_lbls = train_ds.targets.numpy()[:train_size]
    te_imgs = test_ds.data.numpy()[:test_size].reshape(test_size, -1).astype(np.float32) / 255.0
    te_lbls = test_ds.targets.numpy()[:test_size]
    return FlatImageDataset(tr_imgs, tr_lbls), FlatImageDataset(te_imgs, te_lbls)


def load_gender_face(pkl_path: str = None, seed: int = 42):
    """Load the Gender Face dataset from a pickle file.

    The pickle must contain a list [images_array, labels_array] where
    images_array has shape (N, H, W) or (N, D) and labels_array has shape (N,).
    The 85/15 train-test split is reproducible via the seed argument.
    """
    if pkl_path is None:
        pkl_path = DATA_DIR / 'imagestargets.pkl'
    images, labels = pickle.load(open(pkl_path, 'rb'))
    images = images.astype(np.float32)
    images /= max(np.max(np.abs(images)), 1e-5)   # normalise to [-1, 1]
    if images.ndim == 3:
        images = images.reshape(len(images), -1)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(images))
    n_train = int(0.85 * len(images))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    return (
        FlatImageDataset(images[tr_idx], labels[tr_idx]),
        FlatImageDataset(images[te_idx], labels[te_idx]),
    )


def make_loaders(train_ds, test_ds, batch_size: int = 64,
                 batch_size_test: int = 1000, seed: int = 42):
    """Wrap datasets into DataLoader objects with reproducible shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=g, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size_test,
                              shuffle=False)
    return train_loader, test_loader
