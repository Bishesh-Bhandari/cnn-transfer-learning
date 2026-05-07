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
