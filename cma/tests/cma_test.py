# Import modules
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import cma
from sklearn.svm import LinearSVC
import rpy2.robjects as robjects
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from scanpy import read_h5ad
from importlib_resources import files
import subprocess

# Import test data
query = "data/small_test.h5ad"
reference="data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"
od="test_output/"

def test_SVM_prediction():
    command_to_be_executed = ['SVM_prediction', '--reference_H5AD',  str(reference), '--query_H5AD',  str(query), '--LabelsPathTrain',  str(labels), '--OutputDir', str(od)]
    subprocess.check_output(command_to_be_executed, shell=False)
    assert os.path.exists("test_output/SVM_Pred_Labels.csv") == 1
