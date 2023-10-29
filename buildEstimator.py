import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras.models import Sequential, Input
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
import os
import string
import bisect
import re

class BuildEstimator:

    @staticmethod
    def createBlindTestSamples():
        df_train = data[["OverallQual","GrLivArea","GarageCars","FullBath","TotalBsmtSF","YearBuilt","SalePrice"]]
        df_test = test_data[["OverallQual","GrLivArea","GarageCars","FullBath","TotalBsmtSF","YearBuilt"]]

        # Your existing code for creating blind test samples
        data = pd.read_csv("api/lib/data/train.csv")
        test_data = pd.read_csv("api/lib/data/test.csv")
        df = data.drop("Id", axis=1)
        # ... (continue with your data preprocessing steps)

        # Save label encoding information
        labelDict = {}
        leOutput = os.path.join('api/lib/data/labelDict.pickle')
        file = open(leOutput, 'wb')
        pickle.dump(labelDict, file)
        file.close()

        # Save processed data to train.csv and test.csv
        df_train, df_test = train_test_split(df, test_size=0.3)
        df_train.to_csv('api/lib/data/train/csv', index=False)
        df_test.to_csv('api/lib/data/test.csv', index=False)

    @staticmethod
    def getBestPipeline(X, y):
        # Your existing code for getting the best pipeline
        search_params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1]}
        search_params = dict(('estimator__' + k, v) for k, v in search_params.items())

        search_params['normalizer'] = [None, StandardScaler()]
        search_params['featureSelector'] = [None, PCA(n_components=0.90, svd_solver='full')]

        pipe = Pipeline(steps=[
            ('normalizer', None),
            ('featureSelector', None),
            ('estimator', Lasso())
        ])

        cv = GridSearchCV(pipe, search_params, cv=10, verbose=0, scoring='neg_mean_squared_error', n_jobs=-1,
                          error_score=0.0)
        cv.fit(X, y)

        return cv

    @staticmethod
    def createModel():
        # Your existing code for creating and training the neural network
        model = Sequential()
        model.add(Input(shape=(5,)))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(1, activation='relu'))
        print(model.summary())

        scale = StandardScaler()
        X_train = df_train[["Id"]]
        y = df_train[["SalePrice"]]

        # ... (continue with your neural network training code)

        # Save the trained model to flavors_of_cacao.pickle
        file = open('api/lib/model/flavors_of_cacao.pickle', 'wb')
        pickle.dump(model, file)
        file.close()

# Main part of the code
if __name__ == '__main__':
    build_estimator = BuildEstimator()
    build_estimator.createBlindTestSamples()

    df_train = pd.read_csv('api/lib/data/train.csv')
    df_test = pd.read_csv('api/lib/data/test.csv')

    X_fit = df_train.drop(['y'], axis=1)
    y_fit = df_train['y']
    X_blind_test = df_test.drop(['y'], axis=1)
    y_blind_test = df_test['y']

    build_estimator.createModel()
