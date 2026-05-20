
"""
Centralised random-seed control.
Call `set_seed(SEED)` at the top of every training/evaluation script
to guarantee fully reproducible results across all five independent runs
reported in the paper.
"""
import os
import random
import numpy as np
import torch

# Default seed used in all paper experiments
DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix all random-number generators for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic cuDNN behaviour at the cost of some speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
