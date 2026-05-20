
"""
Hyperparameter sensitivity visualisation.

Plots kNN accuracy as a function of each hyperparameter in {a, b, c}
for the VAE+WDIWCD model, holding the other two fixed at optimal values.
Reproduces the sensitivity analysis discussed in Section V of the paper.

Usage
-----
python finetuning/plot_sensitivity.py --output finetuning/results/sensitivity.png
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.seed       import set_seed
from utils.data_utils import load_mnist, make_loaders
from utils.metrics    import knn_accuracy
from classification.models.vae_wdiwcd import VAE_WDIWCD


OPTIMAL = dict(a=0.80, b=0.48, c=0.87)
A_GRID  = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
B_GRID  = [0.1, 0.19, 0.29, 0.39, 0.48, 0.59, 0.69, 0.79, 0.89, 0.99]
C_GRID  = [0.80, 0.82, 0.85, 0.87, 0.90]


def sweep(param: str, grid: list, seed: int = 42):
    set_seed(seed)
    train_ds, test_ds = load_mnist(train_size=5000, test_size=1000)
    input_dim = train_ds[0][0].shape[0]
    accs = []
    for val in grid:
        hp = dict(OPTIMAL)   # copy optimal
        hp[param] = val
        set_seed(seed)
        model = VAE_WDIWCD(input_dim, hidden_dim=500, latent_dim=16,
                           n_classes=10, hist_bins=32)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        tr_loader, _ = make_loaders(train_ds, test_ds, batch_size=64, seed=seed)
        model.train()
        for _ in range(20):
            for x, y in tr_loader:
                opt.zero_grad()
                recon, mu, lv, wl, _, _ = model(x, y)
                loss = VAE_WDIWCD.combined_loss(
                    recon.float(), x, mu, lv, wl,
                    a=hp['a'], b=hp['b'])
                if not torch.isnan(loss): loss.backward(); opt.step()
        model.eval()
        tr_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
        tr_y = np.array([train_ds[i][1] for i in range(len(train_ds))])
        te_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
        te_y = np.array([test_ds[i][1] for i in range(len(test_ds))])
        with torch.no_grad():
            _, z_tr, _, _, _, _ = model(tr_x, torch.tensor(tr_y))
            _, z_te, _, _, _, _ = model(te_x, torch.tensor(te_y))
        accs.append(knn_accuracy(z_tr.numpy(), tr_y, z_te.numpy(), te_y, k=50))
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='finetuning/results/sensitivity.png')
    args = parser.parse_args()
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (param, grid) in zip(axes,
            [('a', A_GRID), ('b', B_GRID), ('c', C_GRID)]):
        accs = sweep(param, grid)
        ax.plot(grid, accs, marker='o', linewidth=2)
        opt_v = OPTIMAL[param]
        ax.axvline(opt_v, color='red', linestyle='--', label=f'optimal={opt_v}')
        ax.set_xlabel(f'Hyperparameter {param}', fontsize=12)
        ax.set_ylabel('kNN Accuracy', fontsize=12)
        ax.set_title(f'Sensitivity to {param}', fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('VAE+WDIWCD Hyperparameter Sensitivity (MNIST)', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved sensitivity plot to {args.output}')


if __name__ == '__main__':
    main()
