import requests
from bs4 import BeautifulSoup
import textwrap

def wrap(text, max_length=80):
    # Split the text into lines
    lines = text.split('\n')
    
    # Process each line
    for i, line in enumerate(lines):
        if len(line) > max_length:
            # The line is too long - wrap it
            lines[i] = '\n'.join(textwrap.wrap(line, max_length))
    
    # Join the processed lines back together
    return '\n'.join(lines)

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