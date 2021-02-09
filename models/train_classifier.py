import sys
from sklearn import metrics
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath:str):
    """
    Load data from db, return text, labels and label names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_data',engine)
    X = df['message']
    y = df.loc[:,'related':]
    
    return X, y, list(y)


stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
def tokenize(text:str)->str:
    """
    Clean the text by keeping only letters and numbers
    The text is also tokenized and lemmatized
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    Build model. The model is tuned using gridsearch.
    The final model is composed by the best combination of 
    estimator and vectorizer setting
    """
    
    pipeline = Pipeline([
        ('vect_tf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    parameters = {
    'vect_tf__max_features':(5000,10000),
    'vect_tf__max_df': (0.5,0.75),
    'clf__estimator': (RidgeClassifier(),LinearSVC(),LogisticRegression(C=1.5))
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the predictions label by label
    Print the classification report
    """
    y_pred = model.predict(X_test)
    for k in range(y_test.shape[1]):
        print(f"Category: {category_names[k]}")
        print(metrics.classification_report(y_test.values[:,k],y_pred[:,k]))


def save_model(model, model_filepath):
    """
    Save model in a pickle file
    """
    with open(model_filepath, 'wb') as output:
            pickle.dump(model, output)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()