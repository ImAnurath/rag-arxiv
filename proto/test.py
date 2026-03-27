import requests

BASE_URL = "http://localhost:8000"

print("Health check:")
print(requests.get(f"{BASE_URL}/health").json())

print("\nQuery test:")
response = requests.post(
    f"{BASE_URL}/query",
    json={"query": "What is RAG in machine learning?", "top_k": 3}
)

print("Status code:", response.status_code)
print("Raw response:", response.text)