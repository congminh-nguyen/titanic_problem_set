# EDA functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Combine train and test data
def combine_train_test(train_df, test_df):
    """
    Combines training and test DataFrames into a single DataFrame with a 'set' column 
    indicating the source of each row.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        The training DataFrame containing the 'Survived' column
    test_df : pandas.DataFrame
        The test DataFrame without the 'Survived' column

    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with an additional 'set' column:
        - 'train' for rows from the training set
        - 'test' for rows from the test set (identified by NaN values in 'Survived' column)
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

# 2. Plots:
# 2.1. Count pairs
def plot_count_pairs(dataframe, columns, hue=None, palette=None):
    """
    Plots count plots for specified columns in a DataFrame with an optional hue.

    :param dataframe: The DataFrame containing the data.
    :param columns: A list of column names to plot.
    :param hue: (Optional) The column name to use as hue.
    :param palette: The color palette to use for the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 4))
        # Reset index to avoid duplicate label issues
        sns.countplot(x=column, data=dataframe.reset_index(drop=True), hue=hue, palette=palette)
        plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
        plt.title(f"Number of passengers / {column}")
        plt.show()

# 2.2. Plot distribution pairs
def plot_distribution(dataframe, column, hue, palette):
    """
    Plots a distribution of a categorical column with a hue.

    :param dataframe: The DataFrame containing the data.
    :param column: The column name to plot.
    :param hue: The column name to use as hue.
    :param palette: A list of colors to use for the hue categories.
    """
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    unique_hues = dataframe[hue].unique()
    
    for i, hue_value in enumerate(unique_hues):
        # Ensure the data selection is correct
        data_to_plot = dataframe.loc[dataframe[hue] == hue_value, column]
        if not data_to_plot.empty:
            sns.histplot(
                data_to_plot,
                color=palette[i],
                ax=ax,
                label=hue_value
            )
    
    ax.set_title(f"Number of passengers / {column}")
    ax.legend(title=hue)
    # Remove the grid
    ax.grid(False)
    plt.show()

# 3. Calculate family size
def calculate_family_size(df):
    """
    Calculate the family size for each passenger in the dataframe.
    
    Args:
    df (pandas.DataFrame): The dataframe containing 'SibSp' and 'Parch' columns.
    
    Returns:
    pandas.Series: A series containing the calculated family size for each passenger.
    """
    return df["SibSp"] + df["Parch"] + 1

# 4. Age interval:
def create_age_intervals(df, column='Age', interval_size=16, max_interval=4):
    """
    Creates age intervals from continuous age data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the age column
    column : str, default='Age'
        Name of the column containing age data
    interval_size : int, default=16
        Size of each age interval
    max_interval : int, default=4
        Number of intervals to create (excluding the first interval for <= interval_size)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with an additional 'Age Interval' column where:
        0: age <= size
        1: size < age <= size*2
        2: size*2 < age <= size*3
        3: size*3 < age <= size*4
        4: age > size*4

    """
    df['Age Interval'] = 0.0
    
    # First interval (0-16)
    df.loc[df[column] <= interval_size, 'Age Interval'] = 0
    
    # Middle intervals
    for i in range(max_interval):
        lower = interval_size * i
        upper = interval_size * (i + 1)
        df.loc[(df[column] > lower) & (df[column] <= upper), 'Age Interval'] = i
    
    # Last interval (> 64)
    df.loc[df[column] > interval_size * max_interval, 'Age Interval'] = max_interval
    
    return df

# 5. Fare interval:
def create_fare_intervals(df, thresholds=[7.91, 14.454, 31], column='Fare'):
    """
    Creates fare intervals based on specified thresholds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the fare column
    thresholds : list, default=[7.91, 14.454, 31]
        List of fare thresholds to create intervals
        Results in intervals (by default):
        0: fare <= 7.91
        1: 7.91 < fare <= 14.454
        2: 14.454 < fare <= 31
        3: fare > 31
    column : str, default='Fare'
        Name of the column containing fare data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with an additional 'Fare Interval' column containing
        interval categories (0-3)

    Example:
    --------
    >>> df = create_fare_intervals(all_df)
    >>> print(df['Fare Interval'].value_counts().sort_index())
    """
    df = df.copy()
    df['Fare Interval'] = 0.0
    
    # First interval
    df.loc[df[column] <= thresholds[0], 'Fare Interval'] = 0
    
    # Middle intervals
    for i in range(len(thresholds)-1):
        df.loc[(df[column] > thresholds[i]) & 
               (df[column] <= thresholds[i+1]), 'Fare Interval'] = i + 1
    
    # Last interval
    df.loc[df[column] > thresholds[-1], 'Fare Interval'] = len(thresholds)
    
    return df

# 6. Create a composed feature: 
def create_composed_feature(df, col1=None, col2=None, new_col=None):
    """
    Creates a new feature combining two columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the columns to be combined
    col1 : str
        Name of the first column to be combined
    col2 : str
        Name of the second column to be combined
    new_col : str, optional
        Name of the new column to be created. If None, it will be "$col1_$col2"

    Returns:
    --------
    pandas.DataFrame
        DataFrame with an additional column combining the two specified columns
        Format: First letter of col1 (uppercase) + '_' + col2 value
        Example: 'F_1' for 'Female' in col1 and '1' in col2

    Example:
    --------
    >>> df = create_composed_feature(train_df, 'Sex', 'Pclass', 'Sex_Pclass')
    >>> print(df['Sex_Pclass'].value_counts())
    """
    if new_col is None:
        new_col = f"{col1}_{col2}"
    
    df[new_col] = df.apply(
        lambda row: f"{row[col1][0].upper()}_{row[col2]}", 
        axis=1
    )
    return df

# Example usage:
# train_df = create_sex_pclass_feature(train_df)