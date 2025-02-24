import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load cleaned data from the specified SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        X (pd.Series): Feature data (messages).
        Y (pd.DataFrame): Target labels (categories).
        category_names (Index): List of category column names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize the input text, applying lemmatization, lower-casing, and stripping.

    Args:
        text (str): The input message string.

    Returns:
        list of str: A list of processed tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline with TF-IDF and MultiOutputClassifier,
    then wrap it in a GridSearchCV to tune hyperparameters.

    Returns:
        GridSearchCV: A scikit-learn GridSearchCV object with the pipeline.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model on the test set, printing out classification reports.

    Args:
        model (GridSearchCV or Pipeline): The trained scikit-learn model.
        X_test (pd.Series): Test features (messages).
        Y_test (pd.DataFrame): True labels for the test set.
        category_names (Index): Names of the target categories.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
        model (GridSearchCV or Pipeline): The trained scikit-learn model to be saved.
        model_filepath (str): The output file path for the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to orchestrate loading data, building model, training,
    evaluating, and saving the classifier.

    Usage:
        python train_classifier.py <database_filepath> <model_filepath>
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data from {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model to {model_filepath}')
        save_model(model, model_filepath)

        print('Model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
