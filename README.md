
# Beyond Correlation: Reproducibility Repository

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-red)](https://arxiv.org/abs/2501.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)

Official reproducibility code for:

> **Beyond Correlation: Learning Supervised, Sample-Distinct, and
> Eigenimage-Interpretable Representation**  
> Mojtaba Moattari · Independent Researcher  
> *arXiv preprint arXiv:2501.XXXXX (2025)*

## Quick links

| Sub-project | Description | Notebook |
|---|---|---|
| [classification/](classification/) | Accuracy simulations (MNIST & Gender-Face) | [Open](classification/notebooks/run_classification.ipynb) |
| [interpretability/](interpretability/) | Eigenimage / eigenface analysis | [Open](interpretability/notebooks/run_interpretability.ipynb) |
| [finetuning/](finetuning/) | Hyperparameter search details | [Open](finetuning/notebooks/run_finetuning.ipynb) |

## Repository structure

```
reproducible_beyond_correlation/
├── classification/
│   ├── models/          # WDIWCD, WDDWCC, VAE, VAE+WDIWCD encoder/decoder
│   ├── utils/           # data loading, metrics, seed control
│   ├── configs/         # YAML configs with optimal hyperparameters
│   ├── figures/         # result figures from the paper
│   └── notebooks/       # step-by-step Jupyter notebook
├── interpretability/
│   ├── models/          # component extractors
│   ├── utils/
│   ├── figures/         # eigenimage & eigenface outputs
│   └── notebooks/
├── finetuning/
│   ├── models/
│   ├── utils/
│   ├── results/         # logged hyperparameter search curves
│   └── notebooks/
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

```bash
git clone https://github.com/moatary/reproducible_beyond_correlation.git
cd reproducible_beyond_correlation
conda env create -f environment.yml
conda activate beyond_corr
# or: pip install -r requirements.txt
```

## Datasets

| Dataset | Source | Auto-download |
|---|---|---|
| MNIST | [LeCun et al.](http://yann.lecun.com/exdb/mnist/) | Yes (torchvision) |
| Gender Face | Preprocessed split provided in `data/` | Yes (script) |

Run `python classification/utils/download_data.py` to fetch and preprocess both datasets.

## Reproducing main results

```bash
# Accuracy simulations (Table 3 / Table 8 in paper)
python classification/train.py --config classification/configs/vae_wdiwcd_optimal.yaml

# Interpretability / eigenimage generation
python interpretability/extract_eigenimages.py --method WDIWCD --dataset mnist

# Hyperparameter sensitivity plots
python finetuning/plot_sensitivity.py
```

## Citation

```bibtex
@article{moattari2025beyond,
  title   = {Beyond Correlation: Learning Supervised, Sample-Distinct, and
             Eigenimage-Interpretable Representation},
  author  = {Moattari, Mojtaba},
  journal = {arXiv preprint arXiv:2501.XXXXX},
  year    = {2025},
  url     = {https://arxiv.org/abs/2501.XXXXX}
}
```

## License
MIT — see [LICENSE](LICENSE).
