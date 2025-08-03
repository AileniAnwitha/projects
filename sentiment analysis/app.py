from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
import pandas as pd
import re
import threading
import webbrowser

app = Flask(__name__)

# Load dataset
try:
    df = pd.read_csv("Amazon_Reviews_Extended_7Reviews.csv", on_bad_lines='skip', encoding='utf-8')
    df = df[['product_name', 'review']]
    df = df.sample(n=100, random_state=42)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")

# Preprocess into cache
product_reviews_cache = {}
for _, row in df.iterrows():
    product = row['product_name'].strip().lower()
    if product not in product_reviews_cache:
        product_reviews_cache[product] = []
    reviews = re.split(r'[.,]\s+', row['review'].strip())
    product_reviews_cache[product].extend([r for r in reviews if r])

# Load sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    product_name = request.form.get("product_name", "").strip().lower()
    
    if product_name not in product_reviews_cache:
        return render_template('result.html', error=f"No reviews found for '{product_name}'.")

    reviews = product_reviews_cache[product_name][:5]
    results = sentiment_pipeline(reviews)

    sentiments = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    detailed = []

    for review, result in zip(reviews, results):
        sentiment = result['label']
        detailed.append({
            "review": review,
            "sentiment": sentiment,
            "emoji": "😃" if sentiment == "POSITIVE" else "😞" if sentiment == "NEGATIVE" else "😐"
        })
        if sentiment in sentiments:
            sentiments[sentiment] += 1
        else:
            sentiments["NEUTRAL"] += 1

    total = sum(sentiments.values())
    percentages = {
        "positive": round(sentiments["POSITIVE"] / total * 100, 2),
        "negative": round(sentiments["NEGATIVE"] / total * 100, 2),
        "neutral": round(sentiments["NEUTRAL"] / total * 100, 2)
    }

    return render_template("result.html",
                           product_name=product_name,
                           reviews=detailed,
                           sentiments=sentiments,
                           percentages=percentages)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)
