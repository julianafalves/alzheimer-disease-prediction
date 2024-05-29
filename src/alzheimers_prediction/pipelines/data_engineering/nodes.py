import pandas as pd
from sklearn.model_selection import train_test_split

def process_gene_data(genetic_data):
    """
    Transpose the DataFrame, replace missing values with the most frequent value, and convert DataFrame to int8 dtype.
    
    Args:
    genetic_data (DataFrame): Input DataFrame.
    
    Returns:
    DataFrame: Processed DataFrame.
    """
    genetic_data = genetic_data.T
    print(genetic_data.head())

    # Handling missing values
    mode_values = genetic_data.mode().iloc[0]
    genetic_data.fillna(mode_values, inplace=True)
    return genetic_data

def consolidate_data(gene_data: pd.DataFrame, phenotypes_numeric: pd.DataFrame):
    """
    Filter data for specific diagnoses, and recode diagnoses into binary labels.
 
    Args:
    gene_data (DataFrame): Gene expression data.
    phenotypes_numeric (DataFrame): Phenotypic data with numeric diagnoses.

    Returns:
    DataFrame: Consolidated DataFrame.
    """
    phenotypes_numeric.set_index(phenotypes_numeric.columns[0], inplace=True)

    
    # Filter and relabel diagnosis
    gene_data = gene_data[phenotypes_numeric['Diag'].isin([1, 3])]
    phenotypes_numeric = phenotypes_numeric[phenotypes_numeric['Diag'].isin([1, 3])]
    phenotypes_numeric[1] = phenotypes_numeric['Diag'].map({1: 0, 3: 1})

    return gene_data, phenotypes_numeric

def split_data(gene_data, phenotypes_numeric, params: dict):
    """
    Split DataFrame into training and testing sets.

    Args:
    df (DataFrame): Input DataFrame.
    params (dict): Parameters for train-test split.

    Returns:
    tuple: X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(gene_data, phenotypes_numeric[1], 
                                                        test_size=params['test_size'], 
                                                        random_state=params['random_state'])
    return X_train, X_test, y_train, y_test
