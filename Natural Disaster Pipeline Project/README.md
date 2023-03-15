# Disaster Response Pipeline Project (Udacity - Data Scientist Nanodegree Program)

## Introduction
This project is part of the Udacity's Data Scientist Nanodegree Program in collaboration with [Figure Eight](https://www.figure-eight.com/).

In this project, the pre-labeled disaster messages will be used to build a disaster response model that can categorize messages received in real time during a disaster event, so that messages can be sent to the right disaster response agency.

This project includes a web application where disaster response worker can input messages received and get classification results.

## File Descriptions
### Folder: app
**run.py** - python script to launch web application.<br/>
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

### Folder: data
**disaster_messages.csv** - real messages sent during disaster events (provided by Figure Eight)<br/>
**disaster_categories.csv** - categories of the messages<br/>
**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ETL pipeline<br/>
**DisasterResponse.db** - cleaned data stored in SQlite database

### Folder: models
**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use<br/>
**classifier.pkl** - pickle file contains trained model<br/>
**ML Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ML pipeline

## Installation
There should be no extra libraries required to install apart from those coming together with Anaconda distribution. There should be no issue to run the codes using Python 3.5 and above.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing an excellent Data Scientist training program.
* [Figure Eight](https://www.figure-eight.com/) for providing dataset to train our model.

