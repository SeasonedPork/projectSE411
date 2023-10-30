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

        # Your existing code for creating blind test samples
        data = pd.read_csv("api/lib/data/train.csv")
        test_data = pd.read_csv("api/lib/data/test.csv")
        df_train = data[["OverallQual","GrLivArea","GarageCars","FullBath","TotalBsmtSF","YearBuilt","SalePrice"]]
        df_test = test_data[["OverallQual","GrLivArea","GarageCars","FullBath","TotalBsmtSF","YearBuilt"]]
        df = data

        # ... (continue with your data preprocessing steps)

        # Save label encoding information
        labelDict = {}
        leOutput = os.path.join('api/lib/data/labelDict.pickle')
        file = open(leOutput, 'wb')
        pickle.dump(labelDict, file)
        file.close()

        # Save processed data to train.csv and test.csv
        df_train, df_test = train_test_split(df, test_size=0.3)
        df_train.to_csv('api/lib/data/train.csv', index=False)
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
        X_train = df_train[["OverallQual","GrLivArea","GarageCars","FullBath","YearBuilt"]]
        y = df_train[["SalePrice"]]
        X_train = scale.fit_transform(X_train)
        y = df_train["SalePrice"].values
        seed = 7
        np.random.seed(seed)
        #split into 67% for train and 33% for test
        #The train_test_split part is a tool that will pick the data and randomize it 
        #it come from the sklearn.model_selection
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, y, test_size=0.33,random_state = seed)
        #Evaluation Process

        # Train:
        loss = 'mse'
        metric = 'mae'
        #accurate epoch is 1750 but due to low CPU use low Epochs
        epochs = 2000
        model.compile(loss=loss, optimizer='adam', metrics=[metric])
        model.fit(X_train, Y_train, epochs=epochs, batch_size=128, verbose=1, validation_data=(X_test, Y_test))

        fit = pd.read_csv('api/lib/data/train.csv')
        Xfit = fit[["OverallQual","GrLivArea","GarageCars","FullBath","YearBuilt"]]
        Yfit = fit[["SalePrice"]]

        blindTest = pd.read_csv('api/lib/data/test.csv')
        XBlindtest = blindTest[["OverallQual","GrLivArea","GarageCars","FullBath","YearBuilt"]]
        print("X_test shape:", X_test.shape)
        print("blindTest shape:", blindTest.shape)
        blindTest["SalePrice"] = model.predict(X_test)
        YBlindtest = blindTest[["SalePrice"]]

        optimizedModel = BuildEstimator.getBestPipeline(Xfit,Yfit).best_estimator_
        yPredFit = optimizedModel.predict(Xfit)
        yPredTest = optimizedModel.predict(XBlindtest)

        fit_score = mean_squared_error(Yfit,yPredFit)
        test_score = mean_squared_error(YBlindtest,yPredTest)
        print("Fit mse = %.2f and test mse - %.2f"%(fit_score,test_score))
        # Save the trained model to flavors_of_cacao.pickle
        file = open('api/lib/model/flavors_of_cacao.pickle', 'wb')
        pickle.dump(optimizedModel, file)
        file.close()

# Main part of the code
if __name__ == '__main__':
    build_estimator = BuildEstimator()
    build_estimator.createBlindTestSamples()

    df_train = pd.read_csv('api/lib/data/train.csv')
    df_test = pd.read_csv('api/lib/data/test.csv')

    build_estimator.createModel()
