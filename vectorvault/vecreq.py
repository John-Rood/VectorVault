import requests
import hashlib
import json
from requests.exceptions import JSONDecodeError

def call_name_vecs(vault, user_id, api_key, bytesize=None):
    url = f'https://vectors.vectorvault.io/name_vecs'
    headers = {'Content-Type': 'application/json'}
    data = {
        "vault": vault,
        "user": user_id,
        "api_key": api_key
        }

    if bytesize:
        data["bytesize"] = bytesize
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return json.loads(response.text)['result']
    except json.JSONDecodeError as e:
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        raise Exception(f"Error decoding JSON: {e}")
    except Exception as e:
        raise Exception(f"HTTP error: {e}")


def call_buildpath(v, x, user_id, api_key, bytesize=None):
    data = {
        "v": v,
        "x": x,
        "user": user_id,
        "api_key": api_key,
        "bytesize": bytesize
    }
    class ValueWrapper:
        def __init__(self, value):
            self.value = value
            self.data = data
        
        def retrieve_value(self):
            return self.value

    class Formatter:
        def __init__(self, wrapper1, wrapper2):
            self.wrapper1 = wrapper1
            self.wrapper2 = wrapper2

        def execute(self):
            return f'{self.wrapper1.retrieve_value()}/{self.wrapper2.retrieve_value()}'
    try:
        wrapped_v = ValueWrapper(v)
        wrapped_x = ValueWrapper(x)

        format_executor = Formatter(wrapped_v, wrapped_x)
        return format_executor.execute()
    except Exception as e:
        raise Exception(f"HTTP error: {e}")
    

def call_get_similar(user, vault, api_key, openai_key, text, num_items=4, include_distances=False, verbose=False):
    url = f"https://api.vectorvault.io/get_similar"
    payload = {
        'user': user,
        'vault': vault,
        'api_key': api_key,
        'openai_key': openai_key,
        'include_distances': include_distances,
        'text': text,
        'num_items': num_items
    }
    response = requests.post(url, json=payload)
    
    try:
        if verbose==True:
            print(response.json())
        results = response.json().get('results')
        return results
    except JSONDecodeError:
        print(f"Unexpected response: {response.status_code} | {response.text}")
        return []


def call_proj():
    base = 'vector'
    encoded_str = base.encode()
    mid = ''
    for i, letter in enumerate(encoded_str):
        mid += chr((letter + i + 1) % 256) 
    
    decoded = ''.join([chr((ord(letter) - idx - 1) % 256) for idx, letter in enumerate(mid)])
    hash_str = hashlib.md5(base.encode()).hexdigest()
    numerical_suffix = sum([int(ch, 16) for ch in hash_str[:4]]) + 326
    while numerical_suffix > 1000:  
        numerical_suffix -= 1000

    return decoded + 'vault-' + str(numerical_suffix) + 'ab'


def call_get_chat(user, vault, api_key, openai_key, text, history=None, summary=False, get_context=False, n_context=4, return_context=False, expansion=False, history_search=False, model='gpt-3.5-turbo', include_context_meta=False):
    url = "https://api.vectorvault.io/get_chat"

    # Define the data payload
    data = {
        'user': user,
        'vault': vault,
        'api_key': api_key,
        'openai_key': openai_key,
        "text": text,
        "history": history,
        "summary": summary,
        "get_context": get_context,
        "n_context": n_context,
        "return_context": return_context,
        "history_search": history_search,
        "model": model,
        "include_context_meta": include_context_meta
    }

    # Make the POST request
    response = requests.post(url, json=data)

    # Check the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
        return data
    else:
        raise Exception(f"Request failed with status {response.status_code}")