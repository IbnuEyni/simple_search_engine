# simple_search_engine

## Overview

This Django application provides a search engine that indexes and queries a predefined set of documents. It uses text processing and machine learning techniques to match queries with these hard-coded documents. The application leverages TfidfVectorizer from scikit-learn for vectorizing documents and querying for similarities.
Features

    Document Querying: Queries predefined documents for similarity.
    Text Preprocessing: Cleans and preprocesses text data for better search accuracy.
    Similarity Calculation: Uses TF-IDF to compute document similarities.

## Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/django-search-engine-app.git
cd django-search-engine-app

## Create a virtual environment:

bash

python -m venv env

## Activate the virtual environment:

    On Windows:

    bash

env\Scripts\activate

On macOS/Linux:

bash

    source env/bin/activate

## Install dependencies:

bash

pip install -r requirements.txt

## Run migrations:

bash

python manage.py migrate

## Run the development server:

bash

    python manage.py runserver

    Access the application:

    Open your web browser and go to http://127.0.0.1:8000/.

# Usage
## Querying Documents

    Navigate to the query page at /query_page/.
    Enter a search query and specify the number of results to return.
    The app will process the query and return the most similar documents from the predefined set.

 # Document Data

The app uses a hard-coded set of documents for querying. These documents are defined in the process_query view function. The predefined documents and their titles are:

    Document Texts:
        "i loved you ethiopia, stored elements in Compress find Sparse Ethiopia is the greatest country in the world of nation at universe"
        "also, sometimes, the same words can have multiple different ‘lemma’s..."
        "With more than million people, ethiopia is the second most populous nation..."
        "The primary purpose of the dam ethiopia is electricity production..."
        "The name that the Blue Nile river loved takes in Ethiopia..."
        "Two non-upgraded loved turbine-generators with MW each are the first loveto go into operation..."

    Titles:
        "Two upgraded"
        "Loved Turbine-Generators"
        "Operation With Loved"
        "National"
        "Power Grid"
        "Generator"

# Text Preprocessing

## The app performs the following preprocessing steps:

    Remove non-ASCII characters
    Remove mentions (e.g., @username)
    Convert text to lowercase
    Remove punctuation and numbers
    Remove extra spaces
    Lemmatize words using nltk

## Dependencies

    Django
    numpy
    pandas
    scikit-learn
    nltk

## Contributing

    Fork the repository.
    Create a new branch (git checkout -b feature-branch).
    Commit your changes (git commit -am 'Add new feature').
    Push to the branch (git push origin feature-branch).
    Create a new Pull Request.


For any questions or issues, please contact shuaibahemdin@gmail.com