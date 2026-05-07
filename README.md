# ShiftGuard10-Robust-Image-Classification
Trained a CIFAR-adapted ResNet-18 in PyTorch with strong augmentations (Crop, flips, ColorJitter, Rotation, Cutout, MixUp α=0.4), label smoothing, SGD + Nesterov momentum, and Cosine Annealing over 100 epochs. Used 4-way Test-Time Augmentation with flip variants for more robust inference.
