import pandas as pd
# This is to create name components from the Name column
def parse_names(row):
    """
    Parse the Name column to extract family name, title, given name, and maiden name.
    Parameters:
        row (pd.Series): A row from the DataFrame containing the Name column.
    Returns:
        pd.Series: A series containing the family name, title, given name, and maiden name.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")

def add_name_components(df):
    """
    Adds columns for parsed name components to the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the names to parse.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional columns for 'Family Name', 'Title', 'Given Name', and 'Maiden Name'.
    """
    # Apply the parse function to each row and assign the results to new columns
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df
    
    