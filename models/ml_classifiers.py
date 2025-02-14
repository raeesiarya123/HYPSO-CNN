import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

# Training time: O(n*d*epochs)
# Prediction time: O(d)
# Space complexity: O(d)
def train_1D_ML_SGD(data_points, annotations, epochs, tolerance, learning_rate, schedule, verbose):
    """
    Train a Stochastic Gradient Descent (SGD) classifier.
    """
    model = SGDClassifier(max_iter=epochs,
                          tol=tolerance,
                          eta0=learning_rate,
                          learning_rate=schedule,
                          verbose=verbose)
    model.fit(data_points, annotations)
    return model

# Training time: O(n*d)
# Prediction time: O(d)
# Space complexity: O(d)
def train_1D_ML_NB(data_points, annotations, class_priors):
    """
    Train a Naive Bayes (NB) classifier.
    """
    model = GaussianNB(priors=class_priors)
    model.fit(data_points, annotations)
    return model

# Training time: O(n*d**2)
# Prediction time: O(d**3)
# Space complexity: O(d**2)
def train_1D_ML_LDA(data_points, annotations, solver, shrinkage):
    """
    Train a Linear Discriminant Analysis (LDA) classifier.
    """
    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    model.fit(data_points, annotations)
    return model

# Training time: O(n*d**2)
# Prediction time: O(d**3)
# Space complexity: O(k*d**2)
def train_1D_ML_QDA(data_points, annotations, reg_param, tolerance):
    """
    Train a Quadratic Discriminant Analysis (QDA) classifier.
    """
    model = QuadraticDiscriminantAnalysis(reg_param=reg_param, tol=tolerance)
    model.fit(data_points, annotations)
    return model