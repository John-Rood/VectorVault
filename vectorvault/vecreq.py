import requests
import json
import sys

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
        print(f"An error occurred: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        sys.exit(1)  
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)  

def call_buildpath(v, x, user_id, api_key, bytesize=None):
    url = f'https://vectors.vectorvault.io/buildpath'
    headers = {'Content-Type': 'application/json'}
    data = {
        "v": v,
        "x": x,
        "user": user_id,
        "api_key": api_key
    }
    if bytesize:
        data["bytesize"] = bytesize
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return json.loads(response.text)['result']
    except json.JSONDecodeError as e:
        print(f"An error occurred: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        sys.exit(1)  
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)  


def call_proj():
    return 'vectorvault-361ab'

def call_get_vaults(user, api_key, vault=None):
    url = f"https://api.vectorvault.io/vaults"
    params = {
        'user': user,
        'vault': vault,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()['vaults']

def call_get_total_vectors(user, vault, api_key):
    url = f"https://api.vectorvault.io/total_vectors"
    params = {
        'user': user,
        'vault': vault,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()['total_vectors']

def call_items_by_vector(user, vault, api_key, vector, num_items=4, include_distances=False):
    url = f"https://api.vectorvault.io/items_by_vector"
    payload = {
        'user': user,
        'vault': vault,
        'vector': vector,
        'include_distances': include_distances,
        'num_items': num_items,
        'api_key': api_key
    }
    response = requests.post(url, json=payload)
    return response.json()['results']

def call_get_similar(user, vault, api_key, text, num_items=4, include_distances=False):
    url = f"https://api.vectorvault.io/get_similar"
    payload = {
        'user': user,
        'vault': vault,
        'api_key': api_key,
        'include_distances': include_distances,
        'text': text,
        'num_items': num_items
    }
    response = requests.post(url, json=payload)
    return response.json()['results']


def call_get_chat(user, vault, api_key, text, history=None, summary=False, get_context=False, n_context=4, return_context=False, expansion=False, history_search=False, model='gpt-3.5-turbo', include_context_meta=False):
    url = "https://api.vectorvault.io/get_chat"

    # Define the data payload
    data = {
        'user': user,
        'vault': vault,
        'api_key': api_key,
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