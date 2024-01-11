from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Function to clean movie titles
def clean(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Function to find movie recommendations
def find_recommendation(n, num_recommendations=5):
    similarity_score = list(enumerate(similarity[n]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i, movie in enumerate(sorted_similar_movies[:num_recommendations]):
        index = movie[0]
        title_from_index = movies_info.loc[movies_info.index == index, 'title'].values[0]
        recommendations.append(title_from_index)

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        result = search(movie_name)

        if not result.empty:
            n = result["index"].iloc[0]
            recommendations = find_recommendation(n)
            return render_template('index.html', movie_name=movie_name, recommendations=recommendations)
        else:
            return render_template('index.html', error_message='No matching movies found.')

    return render_template('index.html')

if __name__ == '__main__':
    # Read movies dataset
    movies_info = pd.read_csv("movies.csv")

    # Add a cleaned title column
    movies_info["clean"] = movies_info["title"].apply(clean)

    # Add an index column
    movies_info["index"] = range(len(movies_info))

    # Vectorize movie titles
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    feature_vectors = vectorizer.fit_transform(movies_info["clean"])

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vectors)

    app.run(debug=True)