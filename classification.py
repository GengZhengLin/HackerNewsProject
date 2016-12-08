"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
import os
from time import time
# import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import svm
import pickle

X_train={}
y_train={}
X_test={}
y_test={}
gname=''
class_weight={}
report={}

def Wite_Report(s='\n'):
    report.write(str(s)+'\n')
    print(s)

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    try:
        Wite_Report('_' * 80)
        Wite_Report("Training: ")
        Wite_Report(clf)

        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        Wite_Report("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        Wite_Report("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        Wite_Report("accuracy:   %0.3f" % score)

        Wite_Report("classification report:")
        Wite_Report(metrics.classification_report(y_test, pred))

        Wite_Report("confusion matrix:")
        Wite_Report(metrics.confusion_matrix(y_test, pred))

        Wite_Report()

        y_scores=[]
        has_roc=True
        if callable(getattr(clf,'predict_proba',None)):
            y_scores=clf.predict_proba(X_test)
        elif callable(getattr(clf,'predict_log_proba',None)):
            y_scores=clf.predict_log_proba(X_test)
        elif callable(getattr(clf,'decision_function',None)):
            y_scores=clf.decision_function(X_test)
        else:
            has_roc=False
            Wite_Report('There is no roc_curve for:'+(str(clf)[0:8]))
        if has_roc:
            roc_data={}
            try:
                roc_data['fpr'],roc_data['tpr'],roc_data['thresh']=metrics.roc_curve(y_test,y_scores)
                with open('reports/'+gname + '_roc_curves/' + (str(clf)[0:8]) + '.plk', 'wb') as f:
                    pickle.dump(roc_data, f, pickle.HIGHEST_PROTOCOL)
                Wite_Report(roc_data['fpr'])
            except:
                Wite_Report('There is no roc_curve for:' + (str(clf)[0:8]))


        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    except:
        print("Unexpected error:", sys.exc_info()[0])

def classification(Xtrain,ytrain,Xtest,ytest,pname,pclass_weight={}):
    global X_train,y_train,X_test,y_test,gname,class_weight,report
    X_train=Xtrain
    y_train=ytrain
    X_test=Xtest
    y_test=ytest
    gname=pname
    class_weight=pclass_weight
    result_directory = 'reports/'
    os.system("taskset -p 0xff %d" % os.getpid())
    print('classification:'+pname)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    roc_directory=result_directory+gname+'_roc_curves/'
    if not os.path.exists(roc_directory):
        os.makedirs(roc_directory)
    plot_directory=result_directory+'plot/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    report=open(result_directory+pname+'-report.txt','wb')

    results = []
    if class_weight:
        results.append(benchmark(svm.SVC(kernel='linear',class_weight=class_weight)))
        return

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(), "Perceptron"),
            (PassiveAggressiveClassifier(), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        Wite_Report('=' * 80)
        Wite_Report(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        Wite_Report('=' * 80)
        Wite_Report("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                                dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

    # Train SGD with Elastic Net penalty
    Wite_Report('=' * 80)
    Wite_Report("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet")))

    # Train NearestCentroid without threshold
    Wite_Report('=' * 80)
    Wite_Report("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    Wite_Report('=' * 80)
    Wite_Report("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))

    Wite_Report('=' * 80)
    Wite_Report("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
      ('classification', LinearSVC())
    ])))


    # make some plots

    # indices = np.arange(len(results))

    # results = [[x[i] for x in results] for i in range(4)]

    # clf_names, score, training_time, test_time = results
    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

    # plt.figure(figsize=(12, 8))
    # plt.title("Score")
    # plt.barh(indices, score, .2, label="score", color='navy')
    # plt.barh(indices + .3, training_time, .2, label="training time",
    #          color='c')
    # plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    # plt.yticks(())
    # plt.legend(loc='best')
    # plt.subplots_adjust(left=.25)
    # plt.subplots_adjust(top=.95)
    # plt.subplots_adjust(bottom=.05)

    # for i, c in zip(indices, clf_names):
    #     plt.text(-.3, i, c)

    # # plt.show()
    # plt.savefig(plot_directory+gname+'.png')
    report.close()
