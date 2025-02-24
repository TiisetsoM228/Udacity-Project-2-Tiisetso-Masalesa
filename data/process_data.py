import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from CSV files, then merge them on 'id'.

    Args:
        messages_filepath (str): Filepath to the messages CSV.
        categories_filepath (str): Filepath to the categories CSV.

    Returns:
        pd.DataFrame: A merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged DataFrame by splitting the categories, converting values
    to binary (0/1), and removing duplicates.

    Args:
        df (pd.DataFrame): The merged DataFrame of messages and categories.

    Returns:
        pd.DataFrame: A cleaned DataFrame with individual binary category columns.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [col.split('-')[0] for col in row]
    categories.columns = category_colnames

    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column] = categories[column].clip(0, 1)

    # Remove the original categories column & concatenate the new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.

    Args:
        df (pd.DataFrame): The cleaned data.
        database_filename (str): Filepath for the output SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to load data, clean data, and save to SQLite database.
    Usage:
        python process_data.py <messages_filepath> <categories_filepath> <database_filepath>
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument.\n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
