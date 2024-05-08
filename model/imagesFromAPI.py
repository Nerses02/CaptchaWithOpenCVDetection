import requests
import os

# Function to download and save an image
def download_image(url, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")

# Add 'model/myAccessKey.txt' file with your actual Unsplash access key
with open("model/myAccessKey.txt", 'r') as file:
    # Read the single line
    access_key = file.readline().strip()

url = 'https://api.unsplash.com/search/photos'
query = 'cat'  # Example query
per_page = 10  # Maximum allowed per page, adjust as per actual API limits
total_pages = 1  # Example to fetch 10 pages

index = 0
for page in range(1, total_pages + 1):
    params = {
        'query': query,
        'client_id': access_key,
        'per_page': per_page,
        'page': page
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        for image in data['results']:
            image_url = f"{image['urls']['regular']}&w=500&h=500&fit=crop"
            filename = f"static/cat/image_{index}.jpg"
            index += 1
            download_image(image_url, filename)
    else:
        print("Failed to fetch images:", response.status_code)
