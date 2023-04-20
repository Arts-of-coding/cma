import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
from sklearn.svm import LinearSVC
import rpy2.robjects as robjects
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from scanpy import read_h5ad

def SVM_prediction(reference_H5AD, query_H5AD, LabelsPathTrain, OutputDir, rejected=False, Threshold_rej=0.7):
    '''
    run baseline classifier: SVM
    Wrapper script to run an SVM classifier with a linear kernel on a benchmark dataset with 5-fold cross validation,
    outputs lists of true and predicted cell labels as csv files, as well as computation time.

    Parameters:
    reference_H5AD, query_H5AD : H5AD files that produce training and testing data,
        cells-genes matrix with cell unique barcodes as row names and gene names as column names.
    LabelsPathTrain : Cell population annotations file path matching the training data (.csv).
    OutputDir : Output directory defining the path of the exported file.
    rejected: If set to True, then the SVMrejected option is chosen. Default: False.
    Threshold_rej : Threshold used when rejecting the cells, default is 0.7.
    '''
    print("Reading in the reference and query H5AD objects")
    training=read_h5ad(reference_H5AD)
    testing=read_h5ad(query_H5AD)
    
    print("Generating training and testing matrices from the H5AD objects")
    
    # training data
    matrix_train = pd.DataFrame.sparse.from_spmatrix(training.X, index=list(training.obs.index.values), columns=list(training.var.features.values))

    # testing data
    try: 
        testing.var['features']
    except KeyError:
        testing.var['features'] = testing.var.index
    
    matrix_test = pd.DataFrame.sparse.from_spmatrix(testing.X, index=list(testing.obs.index.values), columns=list(testing.var.features.values))
    
    print("Unifying training and testing matrices for shared genes")
    
    # subselect the train matrix for values that are present in both
    df_all = training.var[["features"]].merge(testing.var[["features"]].drop_duplicates(), on=['features'], how='left', indicator=True)
    df_all['_merge'] == 'left_only'
    training1 = df_all[df_all['_merge'] == 'both']
    col_one_list = training1['features'].tolist()

    matrix_test = matrix_test[matrix_test.columns.intersection(col_one_list)]
    matrix_train = matrix_train[matrix_train.columns.intersection(col_one_list)]
    matrix_train = matrix_train[list(matrix_test.columns)]
    
    print("Number of genes remaining after unifying training and testing matrices: "+str(len(matrix_test.columns)))
    
    # Convert the ordered dataframes back to nparrays
    matrix_train2 = matrix_train.to_numpy()
    matrix_test2 = matrix_test.to_numpy()
    
    # Delete large objects from memory
    del matrix_train, matrix_test, training, testing
    
    # read the data
    data_train = matrix_train2
    data_test = matrix_test2
    labels_train = pd.read_csv(LabelsPathTrain, header=0,index_col=None, sep=',')
        
    # Set threshold for rejecting cells
    if rejected == True:
        Threshold = Threshold_rej

    print("Log normalizing the training and testing data")
    
    # normalise data
    data_train = np.log1p(data_train)
    data_test = np.log1p(data_test)  
        
    Classifier = LinearSVC()
    pred = []
    
    if rejected == True:
        print("Running SVMrejection")
        clf = CalibratedClassifierCV(Classifier, cv=3)
        probability = [] 
        clf.fit(data_train, labels_train.values.ravel())
        predicted = clf.predict(data_test)
        prob = np.max(clf.predict_proba(data_test), axis = 1)
        unlabeled = np.where(prob < Threshold)
        predicted[unlabeled] = 'Unknown'
        pred.extend(predicted)
        probability.extend(prob)
        pred = pd.DataFrame(pred)
        probability = pd.DataFrame(probability)
        
        # Save the labels and probability
        pred.to_csv(str(OutputDir) + "SVMrej_Pred_Labels.csv", index = False)
        probability.to_csv(str(OutputDir) + "SVMrej_Prob.csv", index = False)
    
    if rejected == False:
        print("Running SVM")
        Classifier.fit(data_train, labels_train.values.ravel())
        predicted = Classifier.predict(data_test)    
        pred.extend(predicted)
        pred = pd.DataFrame(pred)
        
        # Save the predicted labels
        pred.to_csv(str(OutputDir) + "SVM_Pred_Labels.csv", index =False)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run SVM prediction')

    # Add arguments
    parser.add_argument('reference_H5AD', type=str, help='Path to reference H5AD file')
    parser.add_argument('query_H5AD', type=str, help='Path to query H5AD file')
    parser.add_argument('LabelsPathTrain', type=str, help='Path to cell population annotations file')
    parser.add_argument('OutputDir', type=str, help='Path to output directory')

    parser.add_argument('--rejected', dest='rejected', action='store_true', help='Use SVMrejected option')
    parser.add_argument('--Threshold_rej', type=float, default=0.7, help='Threshold used when rejecting cells, default is 0.7')

    # Parse the arguments
    args = parser.parse_args()
    
    # check that output directory exists, create it if necessary
    if not os.path.isdir(args.OutputDir):
        os.makedirs(args.OutputDir)

    # Call the svm_prediction function with the parsed arguments
    SVM_prediction(args.reference_H5AD, args.query_H5AD, args.LabelsPathTrain, args.OutputDir, args.rejected, args.Threshold_rej)

if __name__ == '__main__':
    main()
