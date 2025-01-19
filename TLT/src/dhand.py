# Imports
import os
import pandas as pd
import logging
from typing import Union, Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



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



def save_to_temp_df(
    *dfs: pd.DataFrame,
    dir: str = './temp',
    file_prefix: str = 'test_df'
) -> List[str]:
    """
    Saves multiple DataFrames as CSV files in a specified directory.

    Args:
        *dfs (pd.DataFrame): One or more Pandas DataFrames to save.
        dir (str, optional): Path to the directory where files will be saved. Defaults to './temp'.
        file_prefix (str, optional): Prefix for the filenames. Defaults to 'test_df'.

    Returns:
        List[str]: A list of file paths for the saved CSV files.

    Raises:
        ValueError: If no DataFrames are provided.
        OSError: If the directory cannot be created.
    """
    # Ensure at least one DataFrame is provided
    if not dfs:
        raise ValueError("No DataFrames provided to save.")

    # Ensure the directory exists or create it
    try:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Directory '{dir}' is ready for saving files.")
    except OSError as e:
        raise OSError(f"Failed to create or access directory '{dir}'.") from e

    # Save each DataFrame and collect file paths
    file_paths = []
    for i, df in enumerate(dfs):
        file_name = f"{file_prefix}_{i}.csv"
        file_path = os.path.join(dir, file_name)
        
        try:
            df.to_csv(file_path, index=False)
            logging.info(f"Saved DataFrame to '{file_path}'")
            file_paths.append(file_path)
        except Exception as e:
            logging.error(f"Failed to save DataFrame to '{file_path}': {e}")
            raise

    return file_paths



def clear_files(file_paths: list) -> int:
    """
    Deletes files given a list of file paths.

    Args:
        file_paths (list): A list of file paths to delete.

    Returns:
        int: The number of files successfully deleted.

    Raises:
        ValueError: If the file_paths list is empty.
        FileNotFoundError: If any file in the list does not exist.
        OSError: If there is an issue deleting a file.
    """
    if not file_paths:
        raise ValueError("The file_paths list is empty. Provide at least one file path to delete.")

    deleted_count = 0

    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
                deleted_count += 1
            except Exception as e:
                logging.error(f"Failed to delete file '{file_path}': {e}")
                raise OSError(f"Failed to delete file '{file_path}'.") from e
        else:
            logging.warning(f"File not found: {file_path}")

    logging.info(f"Successfully deleted {deleted_count} file(s).")
    return deleted_count




# Code tests
df = read_source('/home/fdi/AliBagheriNejad/Thesis/MECO/data/vib_case_dataset_ICMS Dataset.xlsx', show_columns=False)

df1,df2  = double_data(
    df,
    ['VibGt_39VS4_1', 'VibGt_39VS4_2'],
)

files = save_to_temp_df(
    df1,
    df2,
    dir = '/home/fdi/AliBagheriNejad/Thesis/MECO/code/TLT/temp'    
)


clear_files(files)




