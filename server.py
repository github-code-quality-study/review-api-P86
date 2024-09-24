import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse, parse_qsl
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

locations = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":            
            query_part = environ['QUERY_STRING']
            params = dict(parse_qsl(query_part))

            location_param = params.get('location')
            start_date = params.get('start_date')
            end_date = params.get('end_date')

            filtered_reviews = reviews.copy()

            if location_param in locations:
                filtered_reviews = list(filter(lambda x: x['Location'] == location_param, filtered_reviews))

            if 'start_date' in params and 'end_date' in params:
                filtered_reviews = list(filter(lambda x: start_date <= x['Timestamp'].split()[0] <= end_date, filtered_reviews))

            elif 'start_date' in params and 'end_date' not in params:
                filtered_reviews = list(filter(lambda x: start_date <= x['Timestamp'].split()[0], filtered_reviews))

            elif 'start_date' not in params and 'end_date' in params:
                filtered_reviews = list(filter(lambda x: x['Timestamp'].split()[0] <= end_date, filtered_reviews))

            # if neither is in the params, skip (no filtering needed)

            # Add sentiment analysis
            for review in filtered_reviews:
                sentiment_scores = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment_scores

            # Sort in descending order by compound value in sentiment
            sorted_filtered_reviews = sorted(filtered_reviews, key=lambda rev: rev['sentiment']['compound'], reverse=True)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(content_length).decode("utf-8")
            params = dict(parse_qsl(request_body))

            if (params.get('Location') not in locations) or ('ReviewBody' not in params):
                start_response("400 Bad Request", [])
                return []
            
            generated_uuid = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            review = {
                'ReviewId': generated_uuid,
                'Location': params['Location'],
                'Timestamp': timestamp,
                'ReviewBody': params['ReviewBody']
            }
            response_body = json.dumps(review, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()