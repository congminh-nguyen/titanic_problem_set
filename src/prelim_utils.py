# Preliminary data inspection functions
import pandas as pd
import numpy as np

# 1. Display Data Overview Function
def display_data_overview(df, name='DataFrame'):
    print(f"--- {name} Head ---")
    display(df.head())
    print(f"\n--- {name} Info ---")
    df.info()
    print(f"\n--- {name} Descriptive Statistics ---")
    display(df.describe())
    print("\n" + "="*50 + "\n")

# 2. Missing Data Analysis Function
def analyse_missing_data(df, name='DataFrame'):
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    types = df.dtypes.astype(str)
    missing_df = pd.DataFrame({
        'Total': total,
        'Percent': percent,
        'Types': types
    })
    missing_df = missing_df.transpose()
    print(f"--- Missing Data in {name} ---")
    display(missing_df)
    print("\n" + "="*50 + "\n")
    return missing_df

# 3. Most Frequent Data Function
def most_frequent_data(df, name='DataFrame'):
    total = df.count()
    most_freq_item = []
    frequency = []
    percent_total = []
    
    for col in df.columns:
        try:
            itm = df[col].value_counts().idxmax()
            val = df[col].value_counts().max()
            pct = round((val / total[col]) * 100, 3)
        except Exception as ex:
            print(f"Error processing column {col}: {ex}")
            itm, val, pct = np.nan, np.nan, np.nan
        most_freq_item.append(itm)
        frequency.append(val)
        percent_total.append(pct)
    
    freq_df = pd.DataFrame({
        'Total': total,
        'Most Frequent Item': most_freq_item,
        'Frequency': frequency,
        'Percent from Total': percent_total
    })
    freq_df = freq_df.transpose()
    print(f"--- Most Frequent Data in {name} ---")
    display(freq_df)
    print("\n" + "="*50 + "\n")
    return freq_df

# 4. Unique Values Analysis Function
def analyse_unique_values(df, name='DataFrame'):
    total = df.count()
    uniques = df.nunique()
    unique_df = pd.DataFrame({
        'Total': total,
        'Uniques': uniques
    })
    unique_df = unique_df.transpose()
    print(f"--- Unique Values in {name} ---")
    display(unique_df)
    print("\n" + "="*50 + "\n")
    return unique_df
