"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        training_set = y.sum()/y.size
        self.probabilities_ = training_set
        return self
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        y = np.random.choice([0, 1], size=(X.shape[0],), p=[1-self.probabilities_, self.probabilities_])
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, t_s=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_s, random_state=i)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_error += (1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True))
        test_error += (1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True))
    train_error /= ntrials
    test_error /= ntrials    
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print 'Plotting...'
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    # #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    # ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    clf_r = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    clf_r.fit(X, y)                  # fit training data using the classifier
    y_pred_r = clf_r.predict(X)        # take the classifier and run it on the training data
    train_error_r = 1 - metrics.accuracy_score(y, y_pred_r, normalize=True)
    print '\t-- training error: %.3f' % train_error_r
    # ### ========== TODO : END ========== ###
    
    
    
    # ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    clf_dt = DecisionTreeClassifier(criterion = "entropy") # create Decision Tree classifier, which includes all model parameters
    clf_dt.fit(X, y)                  # fit training data using the classifier
    y_pred_dt = clf_dt.predict(X)        # take the classifier and run it on the training data
    train_error_dt = 1 - metrics.accuracy_score(y, y_pred_dt, normalize=True)
    print('\t-- training error: %.3f' % train_error_dt)
    # ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    print('Classifying using Majority Vote...')
    train_error, test_error = error(MajorityVoteClassifier(), X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)
    print('\n')
    
    print('Classifying using Random Classifier...')
    train_error, test_error = error(RandomClassifier(), X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)
    print('\n')
    
    print('Classifying using Decision Tree...')
    train_error, test_error = error(DecisionTreeClassifier(criterion = "entropy"), X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)
    print('\n')
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    depth = np.array([i+1 for i in range(20)])
    train_errors = np.zeros(20)
    test_errors= np.zeros(20)
    test_errors_m = np.zeros(20)
    test_errors_r = np.zeros(20)
    for i in range(20):
        train_errors[i], test_errors[i] = error(DecisionTreeClassifier(criterion = "entropy", max_depth=i+1), X, y)
        test_errors_m[i], test_errors_m[i] = error(MajorityVoteClassifier(), X, y)
        test_errors_r[i], test_errors_r[i] = error(RandomClassifier(), X, y)
        # print(i, test_errors[i])
    plt.plot(depth, train_errors, label="Train_error")
    plt.plot(depth, test_errors, label="Test_error")
    plt.plot(depth, test_errors_r, label="Random")
    plt.plot(depth, test_errors_m, label="MajorityVote")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    percentage = [0.05 * i for i in range(1,20)]
    xaxis = [0.1* i for i in range(1,10)]
    train_errors = []
    test_errors= []
    test_errors_r = []
    test_errors_m = []
    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 6)
    clf_r = RandomClassifier()
    clf_m = MajorityVoteClassifier()
    for i in range(1, 20):
        k,q = error(clf, X, y, t_s=1-0.05*i)
        train_errors.append(k)
        test_errors.append(q)

        a,b = error(clf_r, X, y, t_s=1-0.05*i)
        test_errors_r.append(b)

        c,d = error(clf_m, X, y, t_s=1-0.05*i)
        test_errors_m.append(d)

    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    plt.plot(percentage, train_errors, label="Decision_Tree_Train_Error")
    plt.plot(percentage, test_errors, label="Decision_Tree_Test_Error")
    plt.plot(percentage, test_errors_r, label="Random")
    plt.plot(percentage, test_errors_m, label="MajorityVote")
    plt.xticks(xaxis)
    plt.xlabel("Training Set Amount")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
