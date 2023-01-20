import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def splitDf(df, non_X_cols = ['Unnamed: 0','Date','Bond','Return_PD']):

    # normalize all columns which is not in non_X_cols to -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))

    n_rows = 35
    n_groups = df.shape[0] // n_rows

    # Iterate over the groups
    for i in range(n_groups):
        # Get the start and end index of the group
        start_idx = i*n_rows
        end_idx = (i+1)*n_rows
        group_rows = df.iloc[start_idx:end_idx]
    
        # Scale the columns of the group
        for col in group_rows.columns:
            if col not in non_X_cols:
                group_rows[col] = scaler.fit_transform(group_rows[[col]])
        df.iloc[start_idx:end_idx] = group_rows


    # get the unique dates of df_cleaned column: Date
    dates = df['Date'].unique()
    middleDate = dates[round(0.5 * len(dates))]
    seventypercentileDate = dates[round(0.7 * len(dates))]

        # get data until middleDate of df_cleaned
    df_train = df[df['Date'] < middleDate]
    # get data from middleDate of df_cleaned
    df_val = df[df['Date'] >= middleDate]
    df_val = df_val[df_val['Date'] < seventypercentileDate]
    # get data from seventypercentileDate of df_cleaned
    df_test = df[df['Date'] >= seventypercentileDate]

    return df_train, df_val, df_test


def factorX(X):
    X_4factor = X[['Momentum_2_12', 'Pickup', 'Yield', '401_%YoY', 'Beta']]
    X_4factor['Yield'] = X_4factor['Yield'] - X_4factor['401_%YoY']
    X_4factor = X_4factor.drop(columns = ['401_%YoY'])
    return X_4factor

def splitToXY(df, factor, non_X_cols = ['Unnamed: 0','Date','Bond','Return_PD']):

    if (factor == True):
        X = np.zeros((len(df), 4))
    else:
        X = np.zeros((len(df), 85))
    Y = np.zeros((len(df), 1))
    
    df_X = df.drop(columns= non_X_cols)
    df_Y = df['Return_PD']

    if (factor == True):
        df_X = factorX(df_X)

    # Assign df_X to X
    for i in range(len(df_X)):
        X[i] = df_X.iloc[i].values
        Y[i] = df_Y.iloc[i]
        
    return X, Y

def splitAll(df, factor = False):
    df_train, df_val, df_test = splitDf(df)
    X_train, Y_train = splitToXY(df_train, factor)
    X_val, Y_val = splitToXY(df_val, factor)
    X_test, Y_test = splitToXY(df_test, factor)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def bondsNames(df):
    df_bonds = df['Bond'][:35]
    return df_bonds

def evaluate_model(Y_pred, Y_test):
    
    # R squared score
    r2_test = r2_score(Y_test, Y_pred)
    
    return r2_test

def evaluate_model_handwritten(Y_pred, Y_test):
    # R squared score

    num = (sum(pow((Y_test.ravel()-Y_pred),2)))
    denum = sum(pow(Y_test.ravel(),2))

    r2_test = 1-num/denum
    return r2_test

def tuned_model(X_train, Y_train, X_val, Y_val, modelname, params):

    r2_score = -100
    best_model = None

    if modelname == 'OLS':
        best_model = LinearRegression()
        best_model.fit(X_train, Y_train.ravel())

    if modelname == 'Elastic':
        for alpha in params['alpha']:
            for l1 in params['l1_ratio']:
                model = ElasticNet(alpha = alpha, l1_ratio = l1, fit_intercept= False, tol=1).fit(X_train, Y_train.ravel())
                Y_pred = model.predict(X_val)
                r2_test = evaluate_model_handwritten(Y_pred, Y_val)
                if r2_test > r2_score:
                    r2_score = r2_test
                    best_model = model

    if modelname == 'GLM':
        best_model = LinearRegression()
        best_model.fit(X_train, Y_train.ravel())

    if modelname == 'RF':
        for depth in params['max_depth']:
            for features in params['max_features']:
                for n_estimator in params['n_estimators']:
                    model = RandomForestRegressor(max_depth = int(depth), n_estimators = int(n_estimator), max_features = int(features))
                    model.fit(X_train, Y_train.ravel())
                    Y_pred = model.predict(X_val)
                    r2_test = evaluate_model_handwritten(Y_pred, Y_val)
                    if r2_test > r2_score:
                        r2_score = r2_test
                        best_model = model


    if modelname == 'XGB':
        for depth in params['max_depth']:
            for n_estimator in params['n_estimators']:
                for lr in params['learning_rate']:
                    model = XGBRegressor(max_depth = int(depth), n_estimators = int(n_estimator), learning_rate = lr)
                    model.fit(X_train, Y_train.ravel())
                    Y_pred = model.predict(X_val)
                    r2_test = evaluate_model_handwritten(Y_pred, Y_val)
                    if r2_test > r2_score:
                        r2_score = r2_test
                        best_model = model

    if modelname == 'SVM':
        for c in params['C']:
            for gamma in params['gamma']:
                for kernel in params['kernel']:
                    model = SVR(C = c, gamma = gamma, kernel = kernel)
                    model.fit(X_train, Y_train.ravel())
                    Y_pred = model.predict(X_val)
                    r2_test = evaluate_model_handwritten(Y_pred, Y_val)
                    if r2_test > r2_score:
                        r2_score = r2_test
                        best_model = model
                        print(best_model.get_params())
                        

    if modelname == 'NN':
        for hidden_layer_sizes in params['hidden_layer_sizes']:
            for activation in params['activation']:
                for solver in params['solver']:
                    for learning_rate_init in params['learning_rate_init']:
                        for alpha in params['alpha']:
                            for batch_size in params['batch_size']:
                                for learning_rate in params['learning_rate']:
                                    for max_iter in params['max_iter']:
                                        model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver, learning_rate_init = learning_rate_init, alpha = alpha, batch_size = batch_size, learning_rate = learning_rate, max_iter = max_iter)
                                        model.fit(X_train, Y_train)
                                        Y_pred = model.predict(X_val)
                                        r2_test = evaluate_model_handwritten(Y_pred, Y_val)
                                        if r2_test > r2_score:
                                            r2_score = r2_test
                                            best_model = model

                        
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    Y_train_val = np.concatenate((Y_train, Y_val), axis=0)
    best_model.fit(X_train_val, Y_train_val)
    return best_model

def fit_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, modelname, params, retain_month = 1, hypertuneOnce = False):
    print('Starting model fitting...')
    model = tuned_model(X_train, Y_train, X_val, Y_val, modelname = modelname, params = params)
    counter = 0
    Y_pred = np.zeros(Y_test.shape[0])

    print('Starting predictions...')
    for i in range(len(X_test)):
        Y_pred[i] = model.predict(X_test[i].reshape(1, -1))

        counter += 1
        if counter % (retain_month*35) == 0:
            X_train = np.concatenate((X_train, X_val[:35]), axis=0)
            Y_train = np.concatenate((Y_train, Y_val[:35]), axis=0)
            X_val = X_val[35:]
            X_val = np.concatenate((X_val, X_test[:35]), axis=0)
            Y_val = Y_val[35:]
            Y_val = np.concatenate((Y_val, Y_test[:35]), axis=0)

            if (hypertuneOnce):
                model = model.fit(X_train, Y_train)
            else:
                model = tuned_model(X_train, Y_train, X_val, Y_val, modelname = modelname, params = params)

            # print percentage of run time
            print('Percentage of run time: ', round(counter/len(X_test)*100, 2), '%')

    print('Traditional R2 score: ', evaluate_model(Y_pred, Y_test))
    print('Gu Kelly R2 score:', evaluate_model_handwritten(Y_pred, Y_test))

    return Y_pred
