# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the tutorial

A mamba environment `contrastive_learning` is set up with all dependencies:

```bash
mamba activate contrastive_learning
jupyter notebook contrastive_learning_tutorial.ipynb
```

To recreate the environment from scratch:
```bash
mamba create -n contrastive_learning python=3.10 -y
mamba run -n contrastive_learning pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" "torchvision>=0.15.0"
mamba run -n contrastive_learning pip install "numpy>=1.24.0" "matplotlib>=3.7.0" "scikit-learn>=1.2.0" "tqdm>=4.65.0" "ipywidgets>=8.0.0" "Pillow>=9.5.0" jupyter
```

Note: PyTorch must be installed separately from the CPU wheel index because `--index-url` in `requirements.txt` overrides the default PyPI index for all packages.

The notebook is self-contained — all code, explanations, and visualizations are in `contrastive_learning_tutorial.ipynb`. There are no separate Python modules or scripts.

## Architecture overview

The notebook implements **SimCLR** (Chen et al., 2020) on CIFAR-10 in a single sequential flow:

### Core classes (all defined in the notebook)

| Class | Role |
|-------|------|
| `SimCLRAugmentation` | Returns two independently augmented views `(x_i, x_j)` per image |
| `ContrastiveDataset` | Wraps CIFAR-10 to discard labels and return view pairs |
| `SimCLR` | ResNet18 encoder + `ProjectionHead` MLP |
| `NTXentLoss` | NT-Xent loss over a full `(2N × 2N)` cosine similarity matrix |
| `FineTuneClassifier` | Attaches a linear head to a frozen encoder for downstream use |

### Key architectural decisions

- **CIFAR-10 ResNet patch:** The first conv is changed to `kernel_size=3, stride=1, padding=1` and `maxpool` is replaced with `nn.Identity()` — necessary because CIFAR images (32×32) are too small for the standard ImageNet-sized ResNet.
- **Projection head is discarded after pre-training:** Only `model.encoder` is saved/used for downstream evaluation. The projector exists solely to improve pre-training.
- **Checkpoint format:** `simclr_checkpoint.pt` stores both `encoder_state_dict` and `model_state_dict` (full model) plus a `config` dict and final metrics.

### Data flow

```
CIFAR-10 PIL image
  └─ SimCLRAugmentation → (x_i, x_j)        # two augmented views
       └─ SimCLR.forward(x_i, x_j) → (z_i, z_j)  # projected embeddings
            └─ NTXentLoss(z_i, z_j) → scalar loss
```

Evaluation uses frozen encoder features fed into scikit-learn `LogisticRegression` (linear probe) and t-SNE for visualization.

## Environment notes

- Python 3.10 (pinned in `runtime.txt` for Binder)
- CPU-only PyTorch (index URL in `requirements.txt`) — chosen to minimize Binder build time
- Binder badges in `README.md` point to the current branch explicitly; update them when changing branches for cloud sharing
