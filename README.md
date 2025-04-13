# Smart Article Summarizer

A web application that summarizes news articles, highlights key points, and performs sentiment analysis using Natural Language Processing.

## Features

- **Article Summarization**: Get concise summaries of any news article
- **Key Points Extraction**: Automatically identify and highlight important points
- **Sentiment Analysis**: Determine if the article has a positive, negative, or neutral tone
- **User-Friendly Interface**: Simple and intuitive design
- **Paywall Handling**: Fallback mechanisms to handle articles behind paywalls

## Requirements

- Python 3.7+
- Basic dependencies:
  - Flask
  - NLTK
  - BeautifulSoup4
  - Requests
- Advanced features (optional):
  - Transformers (Hugging Face)
  - PyTorch

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd article-summarizer
   ```

2. Run the application using the run.py script, which will guide you through dependency installation:
   ```
   python run.py
   ```

## Usage Options

There are multiple ways to run the application based on your system capabilities:

### 1. Full Version (with AI summarization)
If your system supports PyTorch and Transformers, this will use the Hugging Face pipeline for advanced summarization:
```
python run.py
```

### 2. Basic Version (without AI summarization)
If you have issues installing PyTorch or Transformers, you can still run with basic extractive summarization:
```
python fallback.py
```

### 3. Interactive Setup
The `run.py` script will automatically detect available dependencies and offer appropriate options:
```
python run.py
```

## How It Works

1. **Web Scraping**: The application fetches the article's content using the URL provided
2. **Text Extraction**: It extracts the main text from the HTML using BeautifulSoup
3. **Summarization**: 
   - Full version: Uses Hugging Face Transformers for AI-based abstractive summarization
   - Basic version: Uses extractive summarization (first few sentences and key points)
4. **Key Points Extraction**: Important sentences are identified based on key indicators and relevance
5. **Sentiment Analysis**: NLTK's Sentiment Intensity Analyzer determines the overall sentiment of the article

## Troubleshooting

If you encounter issues with PyTorch installation:

1. Try using the basic version:
   ```
   python fallback.py
   ```

2. Or manually install PyTorch appropriate for your system following the official instructions:
   ```
   # After installing PyTorch correctly
   python app.py
   ```

3. Update your pip and setuptools:
   ```
   pip install --upgrade pip setuptools wheel
   ```

## Limitations

- The application works best with standard news article formats
- Very long articles may be truncated for processing
- The quality of the summary depends on the article's structure and content
- Some websites may block web scraping
- The basic version provides simpler summaries without AI modeling

## Future Improvements

- Add support for PDF articles
- Implement more advanced key points extraction algorithms
- Add multilingual support
- Include topic classification
- Enable saving and comparing multiple article summaries 