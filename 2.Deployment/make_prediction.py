import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from preprocessing import *


clf_path = './Classifiers/'


def predict_proba(testdata):
    
    '''
        Utility function to predict the probability of a transaction being fraudulent.
    '''

    testdata = preprocess(testdata)


    with open(mapping_path+'covariate_shifted_features.pkl', 'rb') as handle:
      features_with_covariate_shift = pickle.load(handle)

    cols = [f for f in testdata if f not in features_with_covariate_shift]

    with open(clf_path+'clf_0.pkl', 'rb') as handle:
        clf_0 = pickle.load(handle)

    with open(clf_path+'clf_1.pkl', 'rb') as handle:
        clf_1 = pickle.load(handle)

    with open(clf_path+'clf_2.pkl', 'rb') as handle:
        clf_2 = pickle.load(handle)

    with open(clf_path+'clf_3.pkl', 'rb') as handle:
        clf_3 = pickle.load(handle)

    with open(clf_path+'clf_4.pkl', 'rb') as handle:
        clf_4 = pickle.load(handle)

    with open(clf_path+'clf_5.pkl', 'rb') as handle:
        clf_5 = pickle.load(handle)

    classifiers = [clf_0, clf_1, clf_2, clf_3, clf_4, clf_5]

    test_proba = np.zeros(len(testdata))

    for clf in classifiers:
        test_proba+=clf.predict_proba(testdata[cols])[:,1]/len(classifiers)

    del clf_0, clf_1, clf_2, clf_3, clf_4, clf_5, features_with_covariate_shift, cols, classifiers
    gc.collect()
    
    return test_proba

