# Contrastive Learning Tutorial: SimCLR from Scratch

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gerhardJaeger/contrastive_learning/HEAD?labpath=contrastive_learning_tutorial.ipynb)
[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/gerhardJaeger/contrastive_learning/HEAD?labpath=contrastive_learning_tutorial.ipynb)

A hands-on tutorial implementing **SimCLR** — a seminal self-supervised contrastive learning framework — using PyTorch on CIFAR-10.

## What you'll learn

- The core intuition behind contrastive self-supervised learning
- How data augmentation drives representation learning
- The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Training a ResNet encoder **without any labels**
- Evaluating learned representations via linear probing and t-SNE

## Contents

| File | Description |
|------|-------------|
| `contrastive_learning_tutorial.ipynb` | Main tutorial notebook (start here) |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python version pin for Binder |

## Quickstart

**In the cloud (Binder):** Click the badge above — no install needed.
> Binder requires this repo to be on a public GitHub host. Push to GitHub first, then the badge will work.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook contrastive_learning_tutorial.ipynb
```

## Notebook outline

1. **The Core Idea** — intuition and the contrastive objective
2. **Setup** — imports and device configuration
3. **Data Augmentation** — the SimCLRAugmentation pipeline with visualization
4. **Dataset Wrapper** — ContrastiveDataset returning view pairs
5. **SimCLR Model** — ResNet18 encoder + MLP projection head
6. **NT-Xent Loss** — implementation with temperature analysis
7. **Training** — full training loop with cosine LR schedule
8. **Linear Probing** — evaluating frozen representations
9. **t-SNE Visualization** — seeing learned structure in 2D
10. **Saving / Loading** — checkpointing and downstream use
11. **Key Takeaways** — what matters and what to try next

## Key concepts

### Contrastive learning at a glance

```
Image x  ──augment t₁──▶  view xᵢ ──▶  Encoder f ──▶  Projector g ──▶  zᵢ ─┐
         ──augment t₂──▶  view xⱼ ──▶  Encoder f ──▶  Projector g ──▶  zⱼ ─┘
                                                                            │
                                                               NT-Xent Loss:
                                                               pull zᵢ ↔ zⱼ together
                                                               push all others apart
```

### NT-Xent Loss

$$\ell(i,j) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

### Linear probe results (CIFAR-10)

| Setting | Test Accuracy |
|---------|--------------|
| Random ResNet18 (baseline) | ~25–35% |
| SimCLR (10 epochs, tutorial) | ~40–60% |
| SimCLR (200 epochs) | ~91% |
| Supervised ResNet18 | ~93% |

## Reference

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). **A Simple Framework for Contrastive Learning of Visual Representations**. ICML 2020. [[arXiv]](https://arxiv.org/abs/2002.05709)
