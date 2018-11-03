import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from gap_statistic import OptimalK

# sklearn
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, validation_curve, GridSearchCV, \
    cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score, log_loss, classification_report, confusion_matrix, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectFromModel

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Nadam
from keras.layers import Dropout
from keras import backend as K
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy


# local items
sys.path.insert(0, os.path.dirname(os.getcwd()))
from utils import *


# Set up directory and file path
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')

php_data_fp = os.path.join(data_dir, 'php8Mz7BG.csv')
sp_data_fp = os.path.join(data_dir, 'dataset_44_spambase.csv')
hwd_data_fp = os.path.join(data_dir, 'dataset_28_optdigits.csv')   # handwritten digits data

# read the data
php_data = pd.read_csv(php_data_fp)
php_X_cols = [col for col in php_data.columns if col != 'Class']
php_X = php_data[php_X_cols]
php_y = php_data['Class']
php_y = php_y.where(php_y==1, other=0)  # change class label 2 to 0

sp_data = pd.read_csv(sp_data_fp)
sp_X_cols = [col for col in sp_data.columns if col != 'class']
sp_X = sp_data[sp_X_cols]
sp_y = sp_data['class']

hwd_data = pd.read_csv(hwd_data_fp)
hwd_X_cols = [col for col in hwd_data.columns if col != 'class']
hwd_X = hwd_data[hwd_X_cols]
hwd_y = hwd_data['class']
