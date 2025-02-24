import json
import plotly
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)  # <-- Now recognized
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# Load model
model = joblib.load("models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """Main page of the web app, generating multiple Plotly visuals."""
    
    # 1. Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2. Top 10 Categories (most frequent)
    # Summing each category column (columns 4 onward are category columns)
    category_sums = df.iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    top_cat_names = list(category_sums.index)
    top_cat_values = category_sums.values

    # 3. Distribution of 'related' category
    related_counts = df['related'].value_counts()
    related_labels = ['Related (1)', 'Not Related (0)']

    # 4. Distribution of 'request' vs 'offer'
    req_off_counts = [
        df['request'].sum(),
        df['offer'].sum()
    ]
    req_off_labels = ['Request', 'Offer']

    # 5. Message Length Distribution (Histogram)
    # Example: count number of words in each message
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    word_counts = df['word_count']

    graphs = [
        # GRAPH 1: Genre Bar Chart
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        # GRAPH 2: Top 10 Categories
        {
            'data': [
                Bar(
                    x=top_cat_names,
                    y=top_cat_values
                )
            ],
            'layout': {
                'title': 'Top 10 Most Frequent Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category", 'tickangle': 35}
            }
        },
        # GRAPH 3: Related Pie Chart
        {
            'data': [
                Pie(
                    labels=related_labels,
                    values=related_counts
                )
            ],
            'layout': {
                'title': 'Distribution of the "related" Category'
            }
        },
        # GRAPH 4: Request vs. Offer
        {
            'data': [
                Pie(
                    labels=req_off_labels,
                    values=req_off_counts
                )
            ],
            'layout': {
                'title': 'Request vs. Offer Messages'
            }
        },
        # GRAPH 5: Word Count Histogram
        {
            'data': [
                {
                    'x': word_counts,
                    'type': 'histogram',
                    'marker': {'color': 'blue'},
                }
            ],
            'layout': {
                'title': 'Distribution of Message Word Counts',
                'xaxis': {'title': "Word Count"},
                'yaxis': {'title': "Number of Messages"},
                'bargap': 0.2
            }
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render `master.html`, passing in the Plotly JSON
    return render_template('master.html', ids=ids, graph_json=graph_json)

@app.route('/go')
def go():
    query = request.args.get('query', '')
    if not query.strip():
        # If query is empty or whitespace, handle gracefully
        return render_template(
            'go.html',
            query=query,
            classification_result={}
        )
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()

