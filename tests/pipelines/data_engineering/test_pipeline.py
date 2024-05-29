"""
This is a boilerplate test file for pipeline 'data_engineering'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
# test_data_processing.py
import pandas as pd
from kedro_datasets.pandas import CSVDataset

from pathlib import Path

def test_diagnoses_values():
    # Load the dataset
    data_catalog = {
        "phenotypic_values": CSVDataset(filepath="data/01_raw/phenotypesNumeric.csv")
    }
    phenotypic_data = data_catalog["phenotypic_values"].load()

    # Check if 'diagnoses' column exists
    assert "Diag" in phenotypic_data.columns, "Column 'diagnoses' not found in the dataset."

    # Extract the 'diagnoses' column
    diagnoses_column = phenotypic_data['Diag']

    # Verify that all values in 'diagnoses' column are 1, 2, or 3
    assert all(diagnoses_column.isin([1, 2, 3])), "Invalid values found in 'diagnoses' column."

