import requests
from bs4 import BeautifulSoup

# URL of the product page
url = "https://www.amazon.com/dp/B08N5WRWNW"

# Headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Fetch the page
response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    # Extract product title
    title = soup.find("span", id="productTitle").get_text(strip=True)
    print(f"Product Title: {title}")
else:
    print(f"Failed to fetch the page. Status Code: {response.status_code}")
