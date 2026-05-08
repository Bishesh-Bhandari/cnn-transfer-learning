 # Imports and device setup — works on Apple M chips, Windows (NVIDIA GPU), and CPU


import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision

from torchvision import datasets, transforms, models

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

from sklearn.preprocessing import label_binarize

import time

import random

import platform

 

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Cross-platform device selection: Apple M (MPS) > NVIDIA GPU (CUDA) > CPU

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda"), f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

        return torch.device("mps"), "Apple M1/M2/M3/M4 (MPS)"

    return torch.device("cpu"), "CPU"

 

device, device_name = get_device()

print(f"Platform: {platform.system()} ({platform.machine()})")

print(f"Using device: {device} ({device_name})")

 

# Safe num_workers for DataLoader: 0 avoids multiprocessing issues on Windows and some Mac setups

NUM_WORKERS = 0

 

# If you see MPS errors on Apple M (e.g. "not implemented"), force CPU: device = torch.device("cpu")
# kernel/filter : a small matrix (eg. 3*3) of numbers .The CNN learns thse number during training and each kernel dte


# Step 1: Define the convolution function (we'll use it in the examples below)

def conv2d_numpy(X, K):

    """Manual 2D convolution — no padding, stride 1.

    X: (H, W) input, K: (kH, kW) kernel. Output: (H - kH + 1, W - kW + 1).

    At each position we multiply the patch with the kernel and sum."""

    H, W = X.shape

    kH, kW = K.shape

    out_h = H - kH + 1

    out_w = W - kW + 1

    Y = np.zeros((out_h, out_w))

    for i in range(out_h):

        for j in range(out_w):

            patch = X[i:i+kH, j:j+kW]

            Y[i, j] = np.sum(patch * K)

    return Y

 

# Step 2: Convolution by hand — 4×4 input, 2×2 kernel → 3×3 output

X_small = np.array([[1., 0., 1., 0.], [0., 1., 1., 0.], [1., 1., 0., 1.], [0., 0., 1., 1.]])

K_small = np.array([[1., -1.], [0., 1.]])

Y_small = conv2d_numpy(X_small, K_small)

print("Input (4×4):\n", X_small)

print("Kernel (2×2):\n", K_small)

print("Output (3×3):\n", Y_small)

print("Top-left output = sum of X[0:2,0:2]*K =", np.sum(X_small[0:2, 0:2] * K_small))

# practical example :edge detector on random 8*8 image  
# the shobel - like kernel  detects vertical edges 

# Practical example: Edge detector on random 8×8 image

# The Sobel-like kernel [[-1,0,1],[-2,0,2],[-1,0,1]] detects vertical edges

np.random.seed(42)

img = np.random.randn(8, 8)

kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # vertical edge

out = conv2d_numpy(img, kernel)

print("Input shape:", img.shape)

print("Kernel shape:", kernel.shape)

print("Output shape:", out.shape)
