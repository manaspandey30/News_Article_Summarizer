import requests

print("Testing simplified API connection...")
try:
    response = requests.post(
        'http://localhost:5000/api/summarize',
        json={'url': 'https://www.bbc.com/news/science-environment-57988023'}
    )
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ API connection successful!")
        data = response.json()
        print("\nSummary:")
        print(data["summary"])
        print("\nKey Points:")
        for i, point in enumerate(data["key_points"], 1):
            print(f"{i}. {point}")
    else:
        print(f"❌ API returned error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error connecting to API: {e}")
    print("Please make sure the Flask server is running on port 5000.") 