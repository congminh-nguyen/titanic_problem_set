#1. Function to handle family type categorization
def categorize_family_type(dfs):
    """
    Categorize family type based on family size.
    """
    for dataset in dfs:
        dataset["Family Type"] = dataset["Family Size"]
        dataset.loc[dataset["Family Size"] == 1, "Family Type"] = "Single"
        dataset.loc[(dataset["Family Size"] > 1) & (dataset["Family Size"] < 5), "Family Type"] = "Small"
        dataset.loc[(dataset["Family Size"] >= 5), "Family Type"] = "Large"
    return dfs

#2. Function to unify titles
def unify_titles(dfs):
    """
    Unify titles to a smaller set of categories.
    """
    for dataset in dfs:
        # Unify 'Miss.'
        dataset['Titles'] = dataset['Titles'].replace(['Mlle.', 'Ms.'], 'Miss.')
        # Unify 'Mrs.'
        dataset['Titles'] = dataset['Titles'].replace('Mme.', 'Mrs.')
        # Unify Rare Titles
        dataset['Titles'] = dataset['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    return dfs

#3. Function to calculate mean survival rates based on Titles and Sex
def calculate_mean_survival(df, group_by_cols):
    """
    Calculate mean survival rates based on a set of columns.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        group_by_cols (list): The columns to group by.
    Returns:
        pd.DataFrame: A DataFrame with the mean survival rates.
    """
    return df[group_by_cols + ['Survived']].groupby(group_by_cols, as_index=False).mean()

#4. Function to map sex to integers
def map_sex_to_int(dfs):
    """
    Map sex to integers.
    """
    for dataset in dfs:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    return dfs

