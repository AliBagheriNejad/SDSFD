# Imports
import os
import pandas as pd
from typing import Union, Optional, List, Tuple

def read_source(
    path: str, 
    show_columns: bool = False
) -> pd.DataFrame:
    """
    Loads a dataset from a given path and optionally displays column names.
    
    Args:
        path (str): The file path of the dataset to load. Supports .csv and .xlsx files.
        show_columns (bool): If True, prints the column names of the dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
        pd.errors.EmptyDataError: If the file is empty.
    """
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at path '{path}' does not exist.")

    # Determine file type and read accordingly
    file_extension = os.path.splitext(path)[-1].lower()
    try:
        if file_extension == ".csv":
            df = pd.read_csv(path)
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Use '.csv' or '.xlsx' files only.")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"The file at '{path}' is empty.") from e

    # Optionally display column names
    if show_columns:
        print(f"\nColumn names of the dataset:\n{list(df.columns)}")

    return df



def double_data(
    df: pd.DataFrame,
    cols: List[str],
    sample_length: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates two new DataFrames from the original DataFrame by splitting 
    the data from specified sensor columns into samples of a fixed length.

    Args:
        df (pd.DataFrame): The source DataFrame containing sensor data.
        cols (List[str]): A list of column names representing sensors. Must contain exactly two column names.
        sample_length (int, optional): Length of each sample. Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, each containing 
        sampled data for the respective sensor column.

    Raises:
        ValueError: If the input DataFrame does not contain the specified columns.
        ValueError: If the `cols` list does not contain exactly two column names.
        ValueError: If `sample_length` is not a positive integer.
    """
    # Validate input columns
    if len(cols) != 2:
        raise ValueError("The 'cols' argument must contain exactly two column names.")
    
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")
    
    # Validate sample_length
    if not isinstance(sample_length, int) or sample_length <= 0:
        raise ValueError("'sample_length' must be a positive integer.")
    
    def create_dataset(sensor_data: pd.Series, sample_length: int) -> pd.DataFrame:
        """
        Splits sensor data into samples of a given length.

        Args:
            sensor_data (pd.Series): A Series containing the sensor's data.
            sample_length (int): Length of each sample.

        Returns:
            pd.DataFrame: A new DataFrame where each row is a sample.
        """
        num_samples = len(sensor_data) // sample_length  # Number of full samples
        # Ensure enough data to create at least one sample
        if num_samples == 0:
            raise ValueError("Insufficient data to create a single sample with the given sample length.")
        # Reshape the data into samples
        samples = sensor_data.iloc[:num_samples * sample_length].values.reshape(-1, sample_length)
        return pd.DataFrame(samples)

    # Create datasets for each sensor
    df_1 = create_dataset(df[cols[0]], sample_length=sample_length)
    df_2 = create_dataset(df[cols[1]], sample_length=sample_length)

    return df_1, df_2


















df = read_source('/home/fdi/AliBagheriNejad/Thesis/MECO/data/vib_case_dataset_ICMS Dataset.xlsx', show_columns=True)

df1,df2  = double_data(
    df,
    ['VibGt_39VS4_1', 'VibGt_39VS4_2'],
)

print (df1.shape)
print (df2.shape)