# Disaster Response Pipeline Project

The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
It also includes a web app where one can test the trained model by inputing a new message and getting a classification result.

## About the data

The data set contains real messages that were sent during disaster events.
There are 26216 messages categorized into 36 different categories.

## File Description
- `app/run.py` : contains the code to run the web app
- `data/process_data.py` : contains the code to clean and store the data into a database
- `models/train_classifier.py` : contains the code to train, tune and save a machine learning classifier


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
