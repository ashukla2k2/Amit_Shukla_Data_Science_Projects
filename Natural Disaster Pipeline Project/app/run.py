import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib

from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


import plotly.express as px

@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df['genre'].value_counts()
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    
    # create visuals
    genre_fig = px.bar(df, x=genre_counts.index, y=genre_counts.values, 
                       labels={'x': 'Genre', 'y': 'Count'}, 
                       title='Distribution of Message Genres')
    
    category_fig = px.bar(x=category_counts.index, y=category_counts.values, 
                          labels={'x': 'Category', 'y': 'Count'},
                          title='Distribution of Message Categories')
    
    # encode plotly graphs in JSON
    genre_graphJSON = genre_fig.to_json()
    category_graphJSON = category_fig.to_json()
    
    # render web page with plotly graphs
    return render_template('master.html', genre_graphJSON=genre_graphJSON, category_graphJSON=category_graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()