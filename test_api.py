import requests

print("Testing API connection...")
try:
    response = requests.post(
        'http://localhost:5000/api/summarize',
        json={'url': 'https://www.bbc.com/news/science-environment-57988023'}
    )
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    print("API connection successful!")
except Exception as e:
    print(f"Error connecting to API: {e}")
    print("Please make sure the Flask server is running on port 5000.") 