import requests

# For a GET request
response = requests.get('http://127.0.0.1:5000/')
print("GET Response Text:", response.text)  # Check the response text

# For a POST request with JSON data
# data = {'key': 'value'}
# response = requests.post('http://127.0.0.1:5000/your-endpoint', json=data)
# print(response.json())  # Assuming the response is in JSON format