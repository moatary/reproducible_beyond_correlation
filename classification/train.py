
"""
Main training script for accuracy simulations.

Usage
-----
python classification/train.py --config classification/configs/vae_wdiwcd_optimal.yaml

Reproduces the classification accuracy results in Table 3 (linear models)
and Table 8 (neural / VAE layer-sharing models) of the paper.
Results are reported as mean ± std over `n_runs` independent seeds.
"""
import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.seed      import set_seed
from utils.data_utils import load_mnist, load_gender_face, make_loaders
from utils.metrics    import knn_accuracy, reconstruction_mse, entropy_score
from models.vae       import VAE, vae_loss
from models.wdiwcd    import WDIWCD
from models.wddwcc    import WDDWCC
from models.vae_wdiwcd import VAE_WDIWCD


MODEL_REGISTRY = {
    'VAE':       VAE,
    'WDIWCD':    WDIWCD,
    'WDDWCC':    WDDWCC,
    'VAE_WDIWCD': VAE_WDIWCD,
}


def build_model(cfg: dict, input_dim: int):
    hp = cfg['hyperparameters']
    name = cfg['model']
    if name == 'VAE':
        return VAE(input_dim, hp.get('hidden_dim', 512), hp.get('latent_dim', 16))
    if name == 'WDIWCD':
        return WDIWCD(input_dim, n_components=hp.get('n_components', 16),
                      hist_bins=hp.get('hist_bins', 32),
                      n_classes=hp.get('n_classes', 10))
    if name == 'WDDWCC':
        return WDDWCC(input_dim, n_components=hp.get('n_components', 16))
    if name == 'VAE_WDIWCD':
        return VAE_WDIWCD(
            input_dim,
            hidden_dim = hp.get('hidden_dim', 500),
            latent_dim = hp.get('latent_dim', 16),
            n_classes  = hp.get('n_classes', 10),
            hist_bins  = hp.get('hist_bins', 32),
            n_heads    = hp.get('n_heads', 1),
        )
    raise ValueError(f'Unknown model: {name}')


def run_one_seed(cfg: dict, seed: int):
    set_seed(seed)
    hp   = cfg['hyperparameters']
    ds   = cfg.get('dataset', 'mnist')

    # Load data
    if ds == 'mnist':
        tr_ds, te_ds = load_mnist()
    else:
        tr_ds, te_ds = load_gender_face(seed=seed)

    input_dim = tr_ds[0][0].shape[0]
    tr_loader, te_loader = make_loaders(
        tr_ds, te_ds,
        batch_size      = hp.get('batch_size', 64),
        batch_size_test = 1000,
        seed            = seed,
    )

    model  = build_model(cfg, input_dim)
    opt    = optim.Adam(model.parameters(), lr=hp.get('learning_rate', 0.001))
    n_iter = hp.get('n_epochs', hp.get('max_iter', 100))
    name   = cfg['model']

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    for epoch in tqdm(range(n_iter), desc=f'[seed={seed}] {name}', leave=False):
        for x_batch, y_batch in tr_loader:
            opt.zero_grad()

            if name in ('WDIWCD', 'WDDWCC'):
                loss = model.compute_loss(
                    x_batch, y_batch,
                    perclass_weight=hp.get('per_class_weight', 0.7),
                )
            elif name == 'VAE':
                recon, mu, log_var, z = model(x_batch)
                loss, _, _ = vae_loss(recon.float(), x_batch, mu, log_var,
                                     beta=hp.get('beta_kld', 1.0))
            elif name == 'VAE_WDIWCD':
                recon, mu, log_var, wdiwcd_loss, _, z = model(x_batch, y_batch)
                loss = VAE_WDIWCD.combined_loss(
                    recon.float(), x_batch, mu, log_var, wdiwcd_loss,
                    a=hp.get('a', 0.80), b=hp.get('b', 0.48),
                )

            if not torch.isnan(loss):
                loss.backward()
                opt.step()

    # ── Evaluation ────────────────────────────────────────────────────────
    model.eval()
    knn_k = cfg.get('evaluation', {}).get('knn_k', 50)

    with torch.no_grad():
        tr_x = torch.stack([tr_ds[i][0] for i in range(len(tr_ds))])
        tr_y = np.array([tr_ds[i][1] for i in range(len(tr_ds))])
        te_x = torch.stack([te_ds[i][0] for i in range(len(te_ds))])
        te_y = np.array([te_ds[i][1] for i in range(len(te_ds))])

        if name == 'VAE':
            _, mu_tr, _, _ = model(tr_x)
            _, mu_te, _, _ = model(te_x)
            recon_te, _, _, _ = model(te_x)
            mse = reconstruction_mse(te_x.numpy(), recon_te.numpy())
        elif name == 'VAE_WDIWCD':
            _, mu_tr, _, _, _, _ = model(tr_x, torch.tensor(tr_y))
            _, mu_te, _, _, _, _ = model(te_x, torch.tensor(te_y))
            recon_te, _, _, _, _, _ = model(te_x, torch.tensor(te_y))
            mse = reconstruction_mse(te_x.numpy(), recon_te.numpy())
        elif name in ('WDIWCD', 'WDDWCC'):
            mu_tr = torch.tensor(model.transform(tr_x))
            mu_te = torch.tensor(model.transform(te_x))
            mse   = None

    acc = knn_accuracy(mu_tr.numpy(), tr_y, mu_te.numpy(), te_y, k=knn_k)
    return {'accuracy': acc, 'mse': mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_runs   = cfg.get('n_runs', 5)
    base_seed = cfg.get('seed', 42)

    all_acc, all_mse = [], []
    for i in range(n_runs):
        result = run_one_seed(cfg, seed=base_seed + i)
        all_acc.append(result['accuracy'])
        if result['mse'] is not None:
            all_mse.append(result['mse'])
        print(f'Run {i+1}/{n_runs}: acc={result["accuracy"]:.4f}', end='')
        if result['mse']:
            print(f'  mse={result["mse"]:.5f}', end='')
        print()

    print(f'\n=== {cfg["model"]} on {cfg["dataset"]} ({n_runs} runs) ===')
    print(f'Accuracy : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}')
    if all_mse:
        print(f'MSE      : {np.mean(all_mse):.5f} ± {np.std(all_mse):.5f}')


if __name__ == '__main__':
    main()
