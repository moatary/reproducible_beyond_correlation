
"""
Eigenimage and eigenface extraction for interpretability analysis.

This script extracts learned weight vectors from each of the proposed models
(WDIWCD, WDDWCC, VAE-NWDIWCD, VAE-NWDDWCC, RLDA), reshapes them into 2-D
image grids, and saves the resulting eigenimage panels as PNG files.

The saved images correspond to the qualitative comparisons shown in Tables 2,
6, and 7 and the figures in Section V-B of the paper.

Background
----------
A model's weight matrix W has shape (input_dim, n_components).  Each column
of W is a projection vector that, when reshaped to (H, W), forms an
*eigenimage* — a spatial template of the patterns the model finds most
informative.  For face datasets these templates are called *eigenfaces*.

Key finding from the paper (Section V-B)
----------------------------------------
WDIWCD eigenimages are sharper and more locally specific than PCA eigenvectors
because the independence criterion allows fine-grained, non-orthogonal structure
to emerge.  WDDWCC eigenimages are less blurred than PCA but less diverse than
WDIWCD, consistent with correlation being a weaker criterion than independence.

Usage
-----
python interpretability/extract_eigenimages.py --method WDIWCD --dataset mnist
python interpretability/extract_eigenimages.py --method WDDWCC --dataset gender_face
python interpretability/extract_eigenimages.py --method VAE_WDIWCD --dataset mnist
python interpretability/extract_eigenimages.py --method VAE_WDDWCC --dataset mnist
python interpretability/extract_eigenimages.py --method RLDA --dataset mnist
python interpretability/extract_eigenimages.py --method PCA --dataset mnist
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.seed       import set_seed
from utils.data_utils import load_mnist, load_gender_face
from classification.models.wdiwcd    import WDIWCD
from classification.models.wddwcc    import WDDWCC
from classification.models.vae_wdiwcd import VAE_WDIWCD


IMG_SIDE = 28   # all datasets resized to 28x28


def extract_weights(method: str, dataset: str,
                    n_components: int = 16,
                    seed: int = 42):
    """Train the specified model and return its weight matrix (input_dim, n_components)."""
    set_seed(seed)
    if dataset == 'mnist':
        train_ds, _ = load_mnist()
    else:
        train_ds, _ = load_gender_face(seed=seed)

    all_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    all_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    input_dim = all_x.shape[1]
    n_classes = int(all_y.max().item()) + 1

    if method == 'PCA':
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(all_x.numpy())
        return pca.components_.T   # (input_dim, n_components)

    if method == 'RLDA':
        # Regularised LDA — project and return loading vectors
        lda = LinearDiscriminantAnalysis(
            n_components=min(n_components, n_classes - 1),
            solver='eigen', shrinkage='auto',
        )
        lda.fit(all_x.numpy(), all_y.numpy())
        W = lda.scalings_  # (input_dim, k)
        # Pad to n_components if k < n_components
        if W.shape[1] < n_components:
            W = np.hstack([W, np.zeros((W.shape[0], n_components - W.shape[1]))])
        return W

    if method == 'WDIWCD':
        from torch.utils.data import DataLoader
        model = WDIWCD(input_dim, n_components=n_components,
                       hist_bins=32, n_classes=n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        for _ in range(50):   # quick demo; use 1000 for paper quality
            for x, y in loader:
                opt.zero_grad()
                loss = model.compute_loss(x, y, perclass_weight=0.7)
                if not torch.isnan(loss): loss.backward(); opt.step()
        return model.w.detach().cpu().numpy()  # (input_dim, n_comp)

    if method == 'WDDWCC':
        from torch.utils.data import DataLoader
        model = WDDWCC(input_dim, n_components=n_components)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        for _ in range(50):
            for x, y in loader:
                opt.zero_grad()
                loss = model.compute_loss(x, y)
                if not torch.isnan(loss): loss.backward(); opt.step()
        return model.w.detach().cpu().numpy()

    if method in ('VAE_WDIWCD', 'VAE_WDDWCC'):
        from torch.utils.data import DataLoader
        model = VAE_WDIWCD(input_dim, hidden_dim=500, latent_dim=16,
                           n_classes=n_classes, hist_bins=32)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        for _ in range(30):
            for x, y in loader:
                opt.zero_grad()
                recon, mu, lv, wl, _, _ = model(x, y)
                loss = VAE_WDIWCD.combined_loss(recon.float(), x, mu, lv, wl,
                                                a=0.80, b=0.48)
                if not torch.isnan(loss): loss.backward(); opt.step()
        # Extract the shared first-layer weights as eigenimage basis
        W = model.encoder.shared_layer.weight.detach().cpu().numpy()  # (H, input)
        # Return first n_components columns after transposing
        return W.T[:, :n_components]

    raise ValueError(f'Unknown method: {method}')


def save_eigenimage_panel(
    W: np.ndarray,
    out_path: str,
    img_side: int = IMG_SIDE,
    title: str = '',
    n_cols: int = 8,
):
    """Reshape weight columns into images and save as a grid panel."""
    n_comp = W.shape[1]
    n_rows = (n_comp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n_comp:
            vec = W[:, i]
            # Normalise to [0, 1] for display
            vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)
            img = vec[:img_side ** 2].reshape(img_side, img_side)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved eigenimage panel: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='WDIWCD',
                        choices=['PCA', 'RLDA', 'WDIWCD', 'WDDWCC',
                                 'VAE_WDIWCD', 'VAE_WDDWCC'])
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'gender_face'])
    parser.add_argument('--n_components', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='interpretability/figures')
    args = parser.parse_args()

    print(f'Extracting {args.method} eigenimages on {args.dataset}...')
    W = extract_weights(args.method, args.dataset,
                        n_components=args.n_components, seed=args.seed)
    out = f'{args.out_dir}/{args.method}_{args.dataset}_eigenimages.png'
    save_eigenimage_panel(W, out,
                          title=f'{args.method} eigenimages — {args.dataset}')


if __name__ == '__main__':
    main()
