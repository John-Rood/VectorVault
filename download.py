import requests
from bs4 import BeautifulSoup

def download_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a valid response

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract text
    text = soup.get_text()

    # Remove leading and trailing whitespace
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text