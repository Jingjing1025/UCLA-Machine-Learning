"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    index = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        for lines in fid:
            extracted_words = extract_words(lines)
            for word in extracted_words:
                if word not in word_list:
                    word_list[word] = index
                    index += 1
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    i = 0

    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        for lines in fid:
            extracted_words = extract_words(lines)
            for word in word_list:
                if word in extracted_words:
                    feature_matrix[i][word_list[word]] = 1
            i += 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    score = 0
    if metric == "accuracy":
        score = metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        score = metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        score = metrics.precision_score(y_true, y_label)
    elif metric == "sensitivity":
        confusion = metrics.confusion_matrix(y_true, y_label)
        score = confusion[1,1]/float(confusion[1,1] + confusion[1,0])
    elif metric == "specificity":
        confusion = metrics.confusion_matrix(y_true, y_label)
        score = confusion[0,0]/float(confusion[0,0] + confusion[0,1])
        
    return score

    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance 
    scores = []   
    for train, test in kf:
        clf.fit(X[train], y[train])
        y_pred = clf.decision_function(X[test])
        perform = performance(y[test], y_pred, metric)
        scores.append(perform)

    score = np.mean(scores)
    return score
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    C = 0
    best_perform = 0
    for val in C_range:
        clf = SVC(kernel='linear', C=val)
        perform = cv_performance(clf, X, y, kf.split(X,y), metric)
        print "C value: ", val, "Performance", perform, ' '
        if perform > best_perform:
            C = val
            best_perform = perform

    return C
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    C = 0
    gamma = 0
    best_perform = 0
    for val_C in C_range:
        for val_gamma in gamma_range:
            clf = SVC(C=val_C, kernel='rbf',gamma=val_gamma)
            perform = cv_performance(clf,X=X, y=y,kf=kf.split(X,y),metric=metric)
            print "C value: ", val_C, "Gamma value: ", val_gamma, "Performance", perform, ' '
            if perform > best_perform:
                max_perform = perform
                C = val_C
                gamma = val_gamma

    return gamma, C
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    train_X = X[:560,:]
    test_X = X[560:,:]
    train_y = y[:560]
    test_y = y[560:]

    # part 2b: create stratified folds (5-fold CV)
    clf = SVC()
    kf = StratifiedKFold(n_splits=5)
    
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    # param_acc = select_param_linear(train_X, train_y, kf, metric = "accuracy")
    # param_f1 = select_param_linear(train_X, train_y, kf, metric = "f1_score")
    # param_auroc = select_param_linear(train_X, train_y, kf, metric = "auroc")
    # param_precision = select_param_linear(train_X, train_y, kf, metric = "precision")
    # param_sensitivity = select_param_linear(train_X, train_y, kf, metric = "sensitivity")
    # param_specificity = select_param_linear(train_X, train_y, kf, metric = "specificity")
    # print param_acc, param_f1, param_auroc, param_precision, param_sensitivity, param_specificity

    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    # param_acc = select_param_rbf(train_X, train_y, kf, metric = "accuracy")
    # param_f1 = select_param_rbf(train_X, train_y, kf, metric = "f1_score")
    # param_auroc = select_param_rbf(train_X, train_y, kf, metric = "auroc")
    # param_precision = select_param_rbf(train_X, train_y, kf, metric = "precision")
    # param_sensitivity = select_param_rbf(train_X, train_y, kf, metric = "sensitivity")
    # param_specificity = select_param_rbf(train_X, train_y, kf, metric = "specificity")
    # print param_acc, param_f1, param_auroc, param_precision, param_sensitivity, param_specificity

    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    # performance = np.zeros((6, 3))
    # for i in range(3):
    #     C_range = 10.0 ** np.arange(-3, 3)
    #     for j in range(C_range.size):
    #         clf = SVC(kernel='linear', C=C_range[j])
    #         accu = cv_performance(clf, train_X, train_y, kf, metric = metric_list[i])
    #         performance[j][i] = round(accu, 4)
    # print performance

    # part 4c: report performance on test data

    # Linear
    print "===== Perfermance Test for Linear Kernel ====="

    clf = SVC(kernel='linear', C=10)
    clf.fit(train_X, train_y)
    print "accuracy = "
    print performance_test(clf, test_X, test_y, metric = "accuracy")
    
    clf = SVC(kernel='linear', C=10)
    clf.fit(train_X, train_y)
    print "f1_score = "
    print performance_test(clf, test_X, test_y, metric = "f1_score")
    
    clf = SVC(kernel='linear', C=1)
    clf.fit(train_X, train_y)
    print "auroc = "
    print performance_test(clf, test_X, test_y, metric = "auroc")

    clf = SVC(kernel='linear', C=10)
    clf.fit(train_X, train_y)
    print "precision = "
    print performance_test(clf, test_X, test_y, metric = "precision")
    
    clf = SVC(kernel='linear', C=0.001)
    clf.fit(train_X, train_y)
    print "sensitivity = "
    print performance_test(clf, test_X, test_y, metric = "sensitivity")
    
    clf = SVC(kernel='linear', C=10)
    clf.fit(train_X, train_y)
    print "specificity = "
    print performance_test(clf, test_X, test_y, metric = "specificity")

    # RBF

    print "===== Perfermance Test for RBF Kernel ====="

    clf = SVC(kernel='rbf', C=100, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "accuracy = "
    print performance_test(clf, test_X, test_y, metric = "accuracy")
    
    clf = SVC(kernel='rbf', C=100, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "f1_score = "
    print performance_test(clf, test_X, test_y, metric = "f1_score")
    
    clf = SVC(kernel='rbf', C=100, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "auroc = "
    print performance_test(clf, test_X, test_y, metric = "auroc")

    clf = SVC(kernel='rbf', C=100, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "precision = "
    print performance_test(clf, test_X, test_y, metric = "precision")
    
    clf = SVC(kernel='rbf', C=0.001, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "sensitivity = "
    print performance_test(clf, test_X, test_y, metric = "sensitivity")
    
    clf = SVC(kernel='rbf', C=100, gamma = 0.01)
    clf.fit(train_X, train_y)
    print "specificity = "
    print performance_test(clf, test_X, test_y, metric = "specificity")

    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
