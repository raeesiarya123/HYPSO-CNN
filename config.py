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
import csv
import random as rd

# Visualization
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns
from PIL import Image

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# Optimizers
from ranger_adabelief import RangerAdaBelief

# HYPSO Package
import hypso

# Progress Bar
from tqdm import tqdm

# Logger
warnings.filterwarnings("ignore")

print("Libraries are loaded!")