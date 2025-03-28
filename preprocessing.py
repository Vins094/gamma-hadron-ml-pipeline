from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
  

def split_data(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42) 
    
    return x_train, x_test, y_train, y_test


def pre_processing():
    # Nested function to fetch data
    def fetch_data():
        from ucimlrepo import fetch_ucirepo
        magic_gamma_telescope = fetch_ucirepo(id=159)
        X = magic_gamma_telescope.data.features
        y = magic_gamma_telescope.data.targets
        df = pd.DataFrame(X)
        df['class'] = y
        return df

    # Nested function for preprocessing
    def process_data(df):
        df = df.drop_duplicates(ignore_index=True)  # Remove duplicates
        dict = {'g': 1, 'h': 0}
        df['class'] = df['class'].map(dict)  # Map class labels
        df = df.drop('fConc1', axis=1)  # Drop unnecessary feature
        
        x = df.drop('class', axis=1)  # Independent variables
        y = df['class']  # Dependent variable

        scaler = StandardScaler()  # Standard scaler
        x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)  # Standardize features
        
        return x_scaled, y

    # Call the nested fetch_data function
    df = fetch_data()

    # Call the nested process_data function
    x, y = process_data(df)
    
    return x, y  # Return the processed features and target labels

def df_to_arr(x_train, x_test, y_train, y_test):
    #Converted to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    return x_train, x_test, y_train, y_test
    



    
    
    
  
   