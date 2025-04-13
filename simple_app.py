from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)
CORS(app)

@app.route('/api/summarize', methods=['POST'])
def summarize_article():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Fetch the article content
        article_content = fetch_article(url)
        
        # Extract the main text
        main_text = extract_text(article_content)
        
        # Generate simple summary
        summary = simple_summarize(main_text)
        
        # Extract key points
        key_points = extract_key_points(main_text)
        
        return jsonify({
            "summary": summary,
            "key_points": key_points,
            "sentiment": {
                "overall": "Neutral",
                "scores": {"compound": 0, "pos": 0.3, "neg": 0.1, "neu": 0.6}
            },
            "url": url,
            "title": extract_title(article_content, url),
            "source": "direct"
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def fetch_article(url):
    """Fetch article content"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    return response.text

def extract_text(html_content):
    """Extract text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get the main article content
    article = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Extract text from paragraphs
    paragraphs = [p.get_text().strip() for p in article if p.get_text().strip()]
    
    # Join paragraphs
    text = ' '.join(paragraphs)
    
    # Clean up text
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_title(html_content, url):
    """Extract title from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find title
    title = soup.find('title')
    if title:
        return title.get_text()
    
    # Try h1 as fallback
    h1 = soup.find('h1')
    if h1:
        return h1.get_text()
    
    return "Article"

def simple_summarize(text, max_chars=500):
    """Very simple summarization - just take the first part of the text"""
    if len(text) <= max_chars:
        return text
    
    # Find the first sentence end after max_chars
    end_pos = text.find('. ', max_chars)
    if end_pos == -1:
        end_pos = max_chars
    else:
        end_pos += 2  # Include the period and space
    
    return text[:end_pos]

def extract_key_points(text, num_points=5):
    """Extract key points - simply the first few sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s) > 20]
    
    # Return at most num_points
    return sentences[:num_points]

if __name__ == '__main__':
    app.run(debug=True) 