from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import json
import os
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# Initialize the sentiment analyzer
try:
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error initializing sentiment analyzer: {str(e)}")
    sia = None

@app.route('/api/summarize', methods=['POST'])
def summarize_article():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Fetch the article content
        article_content, fetch_method = fetch_article(url)
        
        # Extract the main text
        main_text = extract_text(article_content, fetch_method)
        
        # Generate simple summary (first few sentences)
        summary = simple_summarize(main_text)
        
        # Extract key points
        key_points = extract_key_points(main_text)
        
        # Perform sentiment analysis
        sentiment = analyze_sentiment(main_text)
        
        return jsonify({
            "summary": summary,
            "key_points": key_points,
            "sentiment": sentiment,
            "url": url,
            "title": extract_title(article_content, fetch_method, url),
            "source": fetch_method
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_article(url):
    """
    Fetch article content with fallback mechanisms for paywalls
    Returns tuple: (content, method_used)
    """
    # Try direct approach first
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if we got a good amount of content
        content = response.text
        if is_likely_paywall(content):
            raise Exception("Likely paywall detected")
            
        return content, "direct"
    except Exception as first_error:
        # Fallback to Archive.is
        try:
            print(f"Direct fetch failed: {str(first_error)}. Trying archive.is...")
            archive_url = f"https://archive.is/{url}"
            archive_response = requests.get(archive_url, headers=headers, timeout=15)
            if archive_response.status_code == 200:
                return archive_response.text, "archive"
        except Exception as archive_error:
            print(f"Archive fetch failed: {str(archive_error)}")
        
        # Fallback to text extraction API
        try:
            print("Trying extraction API...")
            extract_api_url = "https://extractorapi.com/api/v1/extractor"
            params = {
                "apikey": os.environ.get("EXTRACTOR_API_KEY", "demo"), # Use environment variable or demo key
                "url": url
            }
            api_response = requests.get(extract_api_url, params=params, timeout=15)
            if api_response.status_code == 200:
                return json.dumps(api_response.json()), "api"
        except Exception as api_error:
            print(f"API extraction failed: {str(api_error)}")
            
        # All methods failed
        raise Exception("Could not retrieve article content. The site may have a strict paywall or bot protection.")

def is_likely_paywall(html_content):
    """Check if the content likely contains a paywall"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text().lower()
    
    # Common paywall indicators
    paywall_terms = [
        "subscribe to continue", "subscribe to read", "subscription required",
        "premium article", "to continue reading", "create an account to continue",
        "sign up to read more", "subscribe for full access", "subscribe now",
        "premium content", "paid subscribers only"
    ]
    
    # Check for paywall terms
    if any(term in text for term in paywall_terms):
        return True
    
    # Check for very small content which might indicate restricted access
    paragraphs = soup.find_all('p')
    if len(paragraphs) < 3:
        return True
    
    return False

def extract_text(content, fetch_method):
    """Extract text from content based on fetch method"""
    if fetch_method == "api":
        # Parse JSON from extraction API
        try:
            data = json.loads(content)
            return data.get("text", "")
        except:
            return "Error parsing API response"
    else:
        # Regular HTML parsing
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get the main article content (this may need adjustment based on website structure)
        article = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        # Extract text from paragraphs
        paragraphs = [p.get_text().strip() for p in article if p.get_text().strip()]
        
        # Join paragraphs
        text = ' '.join(paragraphs)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        
        return text

def extract_title(content, fetch_method, original_url):
    """Extract title based on fetch method"""
    if fetch_method == "api":
        try:
            data = json.loads(content)
            return data.get("title", "Article")
        except:
            pass
    
    # Regular HTML parsing
    soup = BeautifulSoup(content, 'html.parser')
    
    # Try to find title in various ways
    title = soup.find('title')
    if title:
        title_text = title.get_text()
        # Clean up title
        site_name = urlparse(original_url).netloc.replace('www.', '').split('.')[0].capitalize()
        title_text = re.sub(r'\s*\|\s*' + site_name + r'.*$', '', title_text)
        title_text = re.sub(r'\s*-\s*' + site_name + r'.*$', '', title_text)
        return title_text
    
    # Try h1 as fallback
    h1 = soup.find('h1')
    if h1:
        return h1.get_text()
    
    return "Article"

def simple_summarize(text, max_sentences=5):
    """
    Simple extractive summarization that takes the first few sentences
    as a summary (often works well for news articles that follow the 
    inverted pyramid structure)
    """
    sentences = sent_tokenize(text)
    
    # If text is short, return as is
    if len(sentences) <= max_sentences:
        return text
    
    # Otherwise take first few sentences (usually contains the gist for news)
    return ' '.join(sentences[:max_sentences])

def extract_key_points(text):
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 5:
        return sentences
    
    try:
        # Get stop words
        stop_words = set(stopwords.words('english'))
        
        # Simple scoring of sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Position score - sentences at the beginning are often important
            position_score = 1.0 / (i + 1)
            
            # Length score - not too short sentences
            length_score = min(1.0, len(sentence.split()) / 20.0)
            
            # Importance indicators score
            indicators = ["important", "significant", "key", "major", "crucial", "essential", 
                         "critical", "vital", "main", "primary", "central", "fundamental"]
            indicator_score = sum(1 for word in sentence.lower().split() if word in indicators) * 0.5
            
            # Named entity score (simplified)
            capitalized_words = sum(1 for word in sentence.split() if word[0].isupper() and word.lower() not in stop_words)
            named_entity_score = min(1.0, capitalized_words / 5.0)
            
            # Combined score with different weights
            total_score = (position_score * 0.4) + (length_score * 0.2) + (indicator_score * 0.2) + (named_entity_score * 0.2)
            sentence_scores.append((i, sentence, total_score))
        
        # Sort sentences by score and take top 5
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
        top_sentences = sorted_sentences[:5]
        
        # Sort selected sentences by their original position
        key_points = sorted(top_sentences, key=lambda x: x[0])
        
        return [sentence for _, sentence, _ in key_points]
    except Exception as e:
        print(f"Error extracting key points: {str(e)}")
        # Return first 5 sentences as fallback
        return sentences[:5]

def analyze_sentiment(text):
    # Check if sentiment analyzer is available
    if sia is None:
        return {
            "overall": "Neutral",
            "scores": {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
        }
    
    try:
        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(text)
        
        # Determine overall sentiment
        if sentiment_scores['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "overall": sentiment,
            "scores": sentiment_scores
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return {
            "overall": "Neutral",
            "scores": {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
        }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Running simplified Article Summarizer (without transformer models)")
    print("This version uses basic extractive summarization techniques.")
    print("="*80 + "\n")
    app.run(debug=True) 