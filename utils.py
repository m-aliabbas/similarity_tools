import pandas as pd
import hashlib


def list_to_df(input_list, columns=['texts']):
    """
    Convert a list to a pandas DataFrame.

    Args:
        input_list (list): The input list.
        columns (list): The column names for the DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame representation of the input list.
    """
    df = pd.DataFrame(input_list, columns=columns)
    return df


def df_to_list(input_df, columns=['texts']):
    """
    Convert a pandas DataFrame to a list.

    Args:
        input_df (pandas.DataFrame): The input DataFrame.
        columns (list): The column names to extract from the DataFrame.

    Returns:
        list: The list representation of the input DataFrame.
    """
    return list(input_df[columns])


def get_object_hash(input_object):
    """
    Calculate the MD5 hash of an input object.

    Args:
        input_object (object): The input object.

    Returns:
        str: The MD5 hash string of the input object.
    """
    if str != type(input_object):
        input_object = str(input_object)
    input_object = input_object.encode()
    hash_string = hashlib.md5(input_object).hexdigest()
    return hash_string
