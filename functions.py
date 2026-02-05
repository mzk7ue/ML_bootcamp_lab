# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% 
def read_data(path):
    """
    Read in CSV data from a URL or file path. 

    Parameters: 
        path (str): The location of the dataset (URL or local file path).
    
    Returns:
        pandas.DataFrame: The raw dataset as a DataFrame.
    """
    return pd.read_csv(path)

# %%
def convert_to_category(df, columns):
    """
    Converts selected columns to 'category' data type.

    Parameters:
        df (pandas.DataFrame): The dataset.
        columns (list): Columns that represent categories, and need to be transformed. 
    
    Returns:
        pandas.DataFrame: The dataset with updated data types.
    """
    df[columns] = df[columns].astype('category')
    return df

# %%
def get_value_counts(df, column): 
    """
    Returns value counts for the specified column. 

    Parameters:
        df (pandas.DataFrame): The dataset.
        column (str): The column name.
    
    Returns:
        pandas.Series: The value counts of the specified column.
    """
    return df[column].value_counts()

# %% 
def standardize(df, columns):
    """
    Standarizes numeric columns whose data is normally distributed or algorithm assumes normal distribution. 

    Parameters:
        df (pandas.DataFrame): The dataset.
        columns (ist): The numeric columns.

    Returns:
        pandas.DataFrame: The dataset with standardized columns
    """
    if isinstance(columns, str):
        columns = [columns]
    df[columns] = StandardScaler().fit_transform(df[columns])
    return df

# %% 
def normalize(df, columns):
    """
    Normalizes numeric columns so that they are on a similar scale, without assuming the normal distribution.

    Parameters:
        df (pandas.DataFrame): The dataset.
        columns (list): The numeric columns.

    Returns:
        pandas.DataFrame: The dataset with normalized numeric columns.
    """
    if isinstance(columns, str):
        columns = [columns]
    df [columns] = MinMaxScaler().fit_transform(df[columns])
    return df

# %%
def drop_columns(df, columns_to_drop):
    """
    Removes columns that are not needed. 

    Parameters:
        df (pandas.DataFrame): The dataset.
        columns_to_drop (list): A list of column names to be removed.
    
    Returns:
        pandas.DataFrame: The dataset with specified columns removed.
    """
    if isinstance(columns_to_drop, str): #accounts for if user passes a single string to columns_to_drop
        columns_to_drop = [columns_to_drop] 
    return df.drop(columns = columns_to_drop, errors = 'ignore')

# %%
def one_hot_encoding(df, columns):
    """
    Converts categorical variables into binary (0/1) indicator columns.

    Parameters:
        df (pandas.DataFrame): The dataset.
        columns (list): The specified categorical columns to encode.

    Returns:
        pandas.DataFrame: The dataset with one-hot encoded variables.
    """
    return pd.get_dummies(df, columns = columns)

# %%
def target_variable(df, new_column, variable, lower, cutoff, upper):
    """
    Creates a binary target variable using a cutoff value.

    Parameters:
        df (pandas.DataFrame): The dataset.
        new_column (str): The name of the new target variable.
        variable (str): The column used to create the target variable.
        lower (float): The minimum value for the bin.
        cutoff (float): The threshold for separating low (0) and high (1).
        upper (float): The maximum value for the bin.

    Returns:
        pandas.DataFrame: The dataset with the new target variable added.
    """
    df[new_column] = pd.cut(df[variable], bins = [lower, cutoff, upper], labels = [0,1])
    return df

# %% 
def prevalence(df, target_col):
    """
    Calculates the prevalence (proportion of positive cases) of the target variable.

    Parameters:
        df (pandas.DataFrame): The dataset.
        target_col (str): The target variable.

    Returns:
        float: The proportion of positive cases (1s).
    """
    return df[target_col].value_counts()[1] / len(df[target_col])

# %% 
def data_partioning(df, train_size, tune_size = 0.5, stratify_col = None):
    """
    Splits the data into training, tuning, and testing sets.

    Parameters: 
        df (pandas.DataFrame): The dataset.
        train_size (int): The size of the training set.
        tune_size (float): The proportion of the remaining data used for tuning. 
        stratify_col (str): The column to stratify splits on.

    Returns:
        tuple: (train_df, tun_df, test_df)
    """
    # If the user wants to stratify by a specific column
    if stratify_col is not None: 
        train, temp = train_test_split(df, train_size = train_size, stratify = df[stratify_col])
        tune, test = train_test_split(temp, train_size = tune_size, stratify = temp[stratify_col])
    else: # If no stratification is needed
        train, temp = train_test_split(df, train_size = train_size)
        tune, test = train_test_split(temp, train_size = tune_size)
    
    return train, tune, test
# %%
