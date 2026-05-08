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
