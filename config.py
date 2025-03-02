# config.py – Common file for all libraries

# System libraries
import os
import sys
import time
import random
import warnings

# Numerical calculations and data processing
import numpy as np
import pandas as pd
import cupy as cp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# HYPSO Package
import hypso

# Logger
warnings.filterwarnings("ignore")

print("Libraries are loaded!")