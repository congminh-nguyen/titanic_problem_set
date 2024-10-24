# EDA functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Combine train and test data
def combine_train_test(train_df, test_df):
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

# 2. Plot count pairs
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
        sns.countplot(x=column, data=dataframe, hue=hue, palette=palette)
        plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
        plt.title(f"Number of passengers / {column}")
        plt.show()

# 3. Plot distribution pairs
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

