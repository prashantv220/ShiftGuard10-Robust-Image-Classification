# ShiftGuard-10 — Robust Image Classification

**Competition:** Shift Guard 10 · Robust Image Classification Challenge (Kaggle)

## What it does

10-class image classifier built from scratch in PyTorch, designed to handle distribution shift at test time. No pretrained weights, no external data — trained entirely on competition data.

## Approach

**Model:** CIFAR-adapted ResNet-18 with 3×3 stem and no maxpool, suited for 32×32 inputs

**Augmentation pipeline:**
- RandomCrop (padding=4, reflect mode), H/V flips, RandomRotation ±15°
- ColorJitter, RandomGrayscale, Cutout (1 hole, 8×8 patch)
- MixUp (α=0.4) — blends training pairs to reduce overconfidence on sharp boundaries

**Training:**
- Loss: Cross-entropy with label smoothing (0.1)
- Optimizer: SGD + Nesterov momentum (lr=0.1, wd=5e-4)
- Scheduler: Cosine Annealing over 100 epochs
- Metric: Macro-F1 on stratified 85/15 train-val split

**Inference:**
- 4-way TTA: original + H-flip + V-flip + both, logits averaged before argmax

## Repository structure

```
├── shiftguard10.ipynb     # Full training and inference notebook
├── submission.csv         # Final predictions
```

## Running it

Designed for Kaggle with GPU. Set `data_path` to your competition directory and run all cells top to bottom. Saves best checkpoint by Macro-F1 during training.

## Key findings

- MixUp was the single biggest robustness gain over vanilla training
- Cutout forced the model to rely on partial views, improving shift tolerance
- L-BFGS not applicable here — SGD + Nesterov generalized better than Adam for this task
- TTA gave consistent improvements with zero training cost

## Dependencies

```
torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow
```
