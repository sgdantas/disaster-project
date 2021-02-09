import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(
              messages_filepath:str,
              categories_filepath:str
              )->pd.DataFrame:
    """
    Load data and return a pandas dataframe with the data merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)   
    df = messages.merge(categories,on=['id'])

    return df


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """
    Clean the data by removing duplicates and separating labels
    Return a pandas dataframe with cleaned data
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column].str.slice(-1))
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    # some labels were 2, fixing it below
    df.related = df.related.map({2:0,0:0,1:1})
    # the child alone is always 0, so it does not contain any useful info
    df.drop(columns=['child_alone'],inplace=True)
    
    return df


def save_data(df:pd.DataFrame, database_filename:str):
    """
    Save the given dataframe in the db with the specified filename
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_data', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()