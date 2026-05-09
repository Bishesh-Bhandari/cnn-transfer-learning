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


# PyTorch Conv2d on a sample image (single channel)

conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)

x = torch.randn(1, 1, 28, 28)  # (batch, channels, H, W)

y = conv(x)

print("Input:", x.shape)

print("Conv2d(1, 4, k=3):", y.shape)

print("Parameters:", sum(p.numel() for p in conv.parameters()))

# learning about the pooling layers 

x = torch.randn(1, 4, 26, 26)

mp = nn.MaxPool2d(2, stride=2)

ap = nn.AvgPool2d(2, stride=2)

print("Input:", x.shape)

print("After MaxPool2d(2):", mp(x).shape   )

print("After AvgPool2d(2):", ap(x).shape)


# full cnn stack :conv-> pool -> fc-> o/p

class TinyCNN(nn.Module):

    """A small CNN for MNIST (28×28 grayscale). Structure: Conv → Pool → Conv → Pool → FC → Output."""

    def __init__(self, num_classes=10):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 1 channel in, 16 filters out; padding=1 keeps size

        self.pool = nn.MaxPool2d(2, 2)                # 2×2 windows, stride 2 → halves each dimension

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 16 channels in, 32 out

        self.fc1 = nn.Linear(32 * 7 * 7, 128)        # After 2 pools: 28→14→7, so 7×7 spatial, 128 nodes (Hidden Layer / fc1)

        self.fc2 = nn.Linear(128, num_classes)

 

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))   # (B,16,14,14)

        x = self.pool(F.relu(self.conv2(x)))   # (B,32,7,7)

        x = x.view(x.size(0), -1)   # Flatten: (B, 32*7*7)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = TinyCNN(10)

print(model)

print("Total params:", sum(p.numel() for p in model.parameters()))


# Manual 1D convolution (NumPy)

def conv1d_numpy(x, w):

    """x: (L,), w: (k,). Output length L - k + 1."""

    L, k = len(x), len(w)

    out = np.zeros(L - k + 1)

    for i in range(len(out)):

        out[i] = np.sum(x[i:i+k] * w)

    return out

 

x = np.array([1., 2., 3., 4., 5., 6., 7.])

w = np.array([0.5, -0.5])

print("1D conv output:", conv1d_numpy(x, w))