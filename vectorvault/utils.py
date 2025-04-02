import requests
from bs4 import BeautifulSoup
import textwrap
import time
from random import randint
from urllib.parse import urljoin

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

def download_url(url, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Add a random delay between requests
            time.sleep(randint(1, 3))
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all links before removing scripts and styles
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Convert relative URLs to absolute
                absolute_url = urljoin(url, href)
                # Store URL along with its anchor text for context
                link_text = a_tag.get_text().strip()
                if link_text and absolute_url:
                    links.append({
                        'url': absolute_url,
                        'text': link_text,
                        'title': a_tag.get('title', '')
                    })
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Return both the text content and the extracted links
            return {
                'content': text,
                'links': links
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_count += 1
                # Exponential backoff - wait longer with each retry
                wait_time = 2 ** retry_count
                print(f"Rate limited. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                # For other HTTP errors, just raise the exception
                raise
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retry_count += 1
            wait_time = 2 ** retry_count
            print(f"Connection error or timeout. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
    
    # If we've exhausted all retries
    raise Exception(f"Failed to download {url} after {max_retries} retries")