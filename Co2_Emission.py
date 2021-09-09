import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to randomly split data into train and test part
from sklearn.preprocessing import StandardScaler # to scale numerical values
from sklearn.impute import SimpleImputer # to impute missing values in Data
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder # to convert categorical features into numberical feature
from sklearn.pipeline import Pipeline # to build a data preprocessing pipeline
from sklearn.compose import ColumnTransformer # to create customize transformer
import os
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Function to load data
def load_data():
    data_root_dir = "E:\\Grras_Data\\Machine_Learning\\Notebooks\\Sachin_Yadav\\datasets"
    data_path = os.path.join(data_root_dir, 'FuelConsumptionCo2.csv')
    fuel = pd.read_csv(data_path)
    X = fuel.drop(['CO2EMISSIONS'], axis=1).copy()
    y = fuel['CO2EMISSIONS']
    return X, y

# Function to calculate RSME score
def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

# Function to generate Model Reports
def model_report(models, X_train, X_test, y_train, y_test):
    for name, model in models:
        val_score = cross_validate(model, X_train, y_train, cv=5, scoring=["neg_mean_squared_error",'r2'])
        val_error = np.sqrt(-val_score['test_neg_mean_squared_error'])
        error = val_error[np.argmin(val_error)]
        val_acc = max(val_score['test_r2'])
        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test  = model.predict(X_test)
        print("_"*80)
        print(f"Report For {name}".center(80))
        print()
        print(f"Training RMSE Error: {rmse(y_train, y_hat_train):.2f}" )
        print(f"Validation    Error: {error:.2f}" )
        print(f"Test     RMSE Error: {rmse(y_test, y_hat_test):.2f}")
        print()
        print(f"Training   Accuracy: {r2_score(y_train, y_hat_train):.2f}")
        print(f"Validation Accuracy: {val_acc:.2f}")
        print(f"Test       Accuracy: {r2_score(y_test, y_hat_test):.2f}")
        print('\n\n')


if __name__ == '__main__':

    X, y = load_data() # Load data
    
    # Splitting data into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Seperating numerical and categorical columns for creating different pipelines
    num_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
                    'FUELCONSUMPTION_COMB_MPG']

    cat_features1 = ['VEHICLECLASS', 'TRANSMISSION']
    cat_features2 = ['FUELTYPE']

    drop_features = ['MODELYEAR', 'MAKE', 'MODEL']

    # Creating pipeline
    final_pipeline = ColumnTransformer([
        # (name, Transformer, column_list)
        ("numerical pipeline", StandardScaler(), num_features),
        ("categorical pipeline1", OrdinalEncoder(), cat_features1),
        ("categorical pipeline2", OneHotEncoder(), cat_features2),
        ('remove features', 'drop', drop_features),
    ])

    # Fitting and transforming train data
    X_train_tr = final_pipeline.fit_transform(X_train)
    
    # Transforming train data
    X_test_tr  = final_pipeline.transform(X_test)
    
    # Creating columns names for making dataframe
    cat_col = list(final_pipeline.named_transformers_['categorical pipeline2'].categories_[0])
    columns = num_features + cat_features1 + list(cat_col)

    # Generating transformed train and test dataframe
    X_train_tr = pd.DataFrame(X_train_tr, columns=columns)
    X_test_tr = pd.DataFrame(X_test_tr, columns=columns)
    
    # Models through which we want our data to be trained 
    models = [
        ('Linear Regression', LinearRegression()),
        ('SGD Regressor', SGDRegressor()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Support Vector Machines', SVR(kernel='linear')),
        ('Random Forest', RandomForestRegressor()),
        ('K-Nearest Neighbors', KNeighborsRegressor())
    ]

    # Generating model reports
    model_report(models, X_train_tr, X_test_tr, y_train, y_test)
          
    
