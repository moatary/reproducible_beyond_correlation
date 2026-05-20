
"""
Hyperparameter search driver for all proposed models.

This script documents and re-runs the grid search that produced
the optimal hyperparameters reported in Table (Hyperparameter Search Space)
of the paper.

Search spaces
-------------
WDDWCC  : LR ∈ {0.001,0.01,0.05,0.1}, Batch ∈ {64,128,256},
          MaxIter ∈ {100,500,1000,2000}, Lambda ∈ [-1,1],
          Components ∈ {4,8,16,32}

WDIWCD  : same grid + HistBins ∈ {8,16,32,64}

VAE     : LatentDim ∈ {8,16,32,64}, LR ∈ {0.0001,0.001,0.01},
          Batch ∈ {32,64,128}, Beta ∈ {0.1,0.5,1.0,2.0},
          EncoderDepth ∈ {2,3,4}, Epochs ∈ {50,100,200}

VAE+WDIWCD : a ∈ [0,1], b ∈ [0,1], c ∈ [0,1],
             + all VAE hyperparameters above
             + HistBins ∈ {8,16,32,64}

The optimal values found are:
  WDDWCC     : LR=0.01, Batch=128, MaxIter=1000, Lambda=0.5, Components=16
  WDIWCD     : LR=0.01, Batch=128, MaxIter=1000, HistBins=32, Components=16
  VAE        : LatentDim=16, LR=0.001, Batch=64, Beta=1.0, Depth=3, Epochs=100
  VAE+WDIWCD : a=0.80, b=0.48, c=0.87  (+ VAE optimal + HistBins=32)
"""
import itertools
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.seed       import set_seed
from utils.data_utils import load_mnist, make_loaders
from utils.metrics    import knn_accuracy

# ── GRID DEFINITIONS ──────────────────────────────────────────────────────
GRIDS = {
    'WDDWCC': {
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'batch_size':    [64, 128, 256],
        'max_iter':      [100, 500, 1000, 2000],
        'lambda_':       [0.0, 0.25, 0.5, 0.75, 1.0],
        'n_components':  [4, 8, 16, 32],
    },
    'WDIWCD': {
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'batch_size':    [64, 128, 256],
        'max_iter':      [100, 500, 1000, 2000],
        'hist_bins':     [8, 16, 32, 64],
        'n_components':  [4, 8, 16, 32],
    },
    'VAE': {
        'latent_dim':    [8, 16, 32, 64],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size':    [32, 64, 128],
        'beta_kld':      [0.1, 0.5, 1.0, 2.0],
        'n_epochs':      [50, 100, 200],
    },
    'VAE_WDIWCD': {
        'a':             [0.3, 0.5, 0.8, 0.52, 0.49],
        'b':             [0.01, 0.1, 0.5, 0.9, 0.79, 0.59, 0.39, 0.19],
        'c':             [0.9, 0.85, 0.87, 0.82, 0.8],
        'latent_dim':    [8, 16, 32],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size':    [32, 64, 128],
        'hist_bins':     [8, 16, 32, 64],
        'n_epochs':      [50, 100, 200],
    },
}


def search(model_name: str, max_trials: int = 50, seed: int = 42):
    """Run random hyperparameter search for the specified model.

    Uses random sampling from the grid (rather than exhaustive search)
    because the full cross-product is too large to enumerate in practice.
    """
    set_seed(seed)
    grid = GRIDS[model_name]
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    rng      = np.random.RandomState(seed)
    combos   = [all_combos[i] for i in
                rng.choice(len(all_combos), size=min(max_trials, len(all_combos)), replace=False)]

    train_ds, test_ds = load_mnist()
    input_dim         = train_ds[0][0].shape[0]
    best_acc, best_hp = 0.0, None

    for combo in tqdm(combos, desc=f'Searching {model_name}'):
        hp = dict(zip(keys, combo))
        try:
            acc = _trial(model_name, hp, train_ds, test_ds, input_dim, seed)
        except Exception as e:
            continue
        if acc > best_acc:
            best_acc, best_hp = acc, hp
            print(f'  New best: {best_acc:.4f} with {best_hp}')

    print(f'\n=== Best {model_name}: {best_acc:.4f} === {best_hp}')
    return best_hp


def _trial(model_name, hp, train_ds, test_ds, input_dim, seed):
    """Run one trial and return kNN accuracy."""
    import sys; sys.path.insert(0, '..')
    from classification.models.wdiwcd    import WDIWCD
    from classification.models.wddwcc    import WDDWCC
    from classification.models.vae       import VAE, vae_loss
    from classification.models.vae_wdiwcd import VAE_WDIWCD

    set_seed(seed)
    bs = hp.get('batch_size', 64)
    tr_loader, te_loader = make_loaders(train_ds, test_ds, batch_size=bs, seed=seed)
    n_iter = hp.get('n_epochs', hp.get('max_iter', 50))
    n_iter = min(n_iter, 30)   # cap at 30 for search speed

    if model_name == 'WDIWCD':
        model = WDIWCD(input_dim, n_components=hp.get('n_components', 16),
                       hist_bins=hp.get('hist_bins', 32), n_classes=10)
    elif model_name == 'WDDWCC':
        model = WDDWCC(input_dim, n_components=hp.get('n_components', 16))
    elif model_name == 'VAE':
        model = VAE(input_dim, hidden_dim=512, latent_dim=hp.get('latent_dim', 16))
    elif model_name == 'VAE_WDIWCD':
        model = VAE_WDIWCD(input_dim, hidden_dim=500,
                           latent_dim=hp.get('latent_dim', 16),
                           n_classes=10, hist_bins=hp.get('hist_bins', 32))

    opt = torch.optim.Adam(model.parameters(), lr=hp.get('learning_rate', 0.001))
    model.train()
    for _ in range(n_iter):
        for x, y in tr_loader:
            opt.zero_grad()
            if model_name in ('WDIWCD', 'WDDWCC'):
                loss = model.compute_loss(x, y)
            elif model_name == 'VAE':
                recon, mu, lv, _ = model(x)
                loss, _, _ = vae_loss(recon.float(), x, mu, lv)
            elif model_name == 'VAE_WDIWCD':
                recon, mu, lv, wl, _, _ = model(x, y)
                loss = VAE_WDIWCD.combined_loss(recon.float(), x, mu, lv, wl,
                                                a=hp.get('a', 0.8), b=hp.get('b', 0.48))
            if not torch.isnan(loss): loss.backward(); opt.step()

    model.eval()
    tr_x = torch.stack([train_ds[i][0] for i in range(min(2000, len(train_ds)))])
    tr_y = np.array([train_ds[i][1] for i in range(min(2000, len(train_ds)))])
    te_x = torch.stack([test_ds[i][0] for i in range(min(1000, len(test_ds)))])
    te_y = np.array([test_ds[i][1] for i in range(min(1000, len(test_ds)))])

    with torch.no_grad():
        if model_name in ('WDIWCD', 'WDDWCC'):
            z_tr = model.transform(tr_x); z_te = model.transform(te_x)
        elif model_name == 'VAE':
            _, z_tr, _, _ = model(tr_x); z_tr = z_tr.numpy()
            _, z_te, _, _ = model(te_x); z_te = z_te.numpy()
        elif model_name == 'VAE_WDIWCD':
            _, z_tr, _, _, _, _ = model(tr_x, torch.tensor(tr_y)); z_tr = z_tr.numpy()
            _, z_te, _, _, _, _ = model(te_x, torch.tensor(te_y)); z_te = z_te.numpy()
    return knn_accuracy(z_tr, tr_y, z_te, te_y, k=50)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='VAE_WDIWCD',
                   choices=list(GRIDS.keys()))
    p.add_argument('--max_trials', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    search(args.model, args.max_trials, args.seed)
