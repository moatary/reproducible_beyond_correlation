
"""
Evaluation metrics used throughout the paper.

Metrics
-------
knn_accuracy       : k-Nearest-Neighbour classification in latent space.
reconstruction_mse : Mean squared error between input and VAE reconstruction.
entropy_score      : Shannon entropy of projected feature histograms
                     (higher = more diverse / higher contrast).
amari_index        : Source-separation quality index (lower = better).
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knn_accuracy(z_train: np.ndarray, y_train: np.ndarray,
                 z_test: np.ndarray,  y_test: np.ndarray,
                 k: int = 50) -> float:
    """Compute k-NN accuracy in the learned latent space."""
    scaler = StandardScaler()
    z_tr = scaler.fit_transform(z_train)
    z_te = scaler.transform(z_test)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(z_tr, y_train)
    return float(clf.score(z_te, y_test))


def reconstruction_mse(x_orig: np.ndarray, x_recon: np.ndarray) -> float:
    """Mean squared reconstruction error, averaged over all pixels and samples."""
    return float(np.mean((x_orig - x_recon) ** 2))


def entropy_score(projections: np.ndarray, n_bins: int = 64) -> float:
    """Shannon entropy of the 1-D marginal histogram of projected features."""
    hist, _ = np.histogram(projections.ravel(), bins=n_bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist + 1e-12)))


def amari_index(W: np.ndarray, A: np.ndarray) -> float:
    """Amari performance index for ICA source separation quality.

    Parameters
    ----------
    W : estimated unmixing matrix  (n_sources x n_sources)
    A : true mixing matrix         (n_sources x n_sources)
    """
    P = W @ A
    n = P.shape[0]
    err  = sum(np.sum(np.abs(P[i]) / np.max(np.abs(P[i])) - 1) for i in range(n))
    err += sum(np.sum(np.abs(P[:, j]) / np.max(np.abs(P[:, j])) - 1) for j in range(n))
    return float(err / (2 * n * (n - 1)))


def run_n_seeds(fn, n_seeds: int = 5, base_seed: int = 42):
    """Run *fn(seed)* for n_seeds different seeds; return (mean, std) of results."""
    results = [fn(base_seed + i) for i in range(n_seeds)]
    return float(np.mean(results)), float(np.std(results))
