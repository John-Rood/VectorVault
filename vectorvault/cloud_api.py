import requests
import hashlib
import json
from requests.exceptions import JSONDecodeError

API_BASE_URL = "https://api.vectorvault.io" 
VECTOR_BASE_URL = "https://vectors.vectorvault.io" 
access_token = None
refresh_token = None

def call_name_vecs(vault, user_id, api_key, bytesize=None):
    url = f'{VECTOR_BASE_URL}/name_vecs'
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
    

def call_req(user, api_key, num=4):
    url = f"{API_BASE_URL}/make_request"
    payload = {
        'user': user,
        'api_key': api_key,
        'num': num,
    }
    response = requests.post(url, json=payload)
    
    try:
        return response.json()
    except JSONDecodeError:
        print(f"Unexpected response: {response.status_code} | {response.text}")
        return None


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


def login_with_api(email, api_key):
    global access_token, refresh_token
    url = f"{API_BASE_URL}/login_with_api"
    payload = {
        "email": email,
        "api_key": api_key
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        access_token = result['access_token']
        refresh_token = result.get('refresh_token')
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error during login_with_api: {e}")
        return False


def call_cloud_save(user, api_key, vault, embeddings_model, text, meta=None, name=None, split=None, split_size=None):
    global access_token
    # Authenticate using API key if access_token is None
    if access_token is None:
        success = login_with_api(user, api_key)
        if not success:
            print("Authentication failed.")
            return None

    url = f"{API_BASE_URL}/add_cloud"
    data = {
        "vault": vault,
        "embeddings_model": embeddings_model,
        "text": text,
        "meta": meta,
        "name": name,
        "split": split,
        "split_size": split_size,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error during call_cloud_save: {e}")
        return None
    

def run_flow(user, api_key, flow_name, message, history='', vault='home', conversation_user_id=None):
    global access_token
    # Authenticate using API key if access_token is None
    if access_token is None:
        success = login_with_api(user, api_key)
        if not success:
            print("Authentication failed.")
            return None
    url = f"{API_BASE_URL}/flow"
    payload = {
        "flow_id": flow_name,
        "message": message,
        "history": history,
        "vault": vault,
        "conversation_user_id": conversation_user_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get('response', None)
    except requests.exceptions.RequestException as e:
        print(f"Error during run_flow: {e}")
        return None


def run_flow_stream(user, api_key, flow_name, message, history='', vault='home', conversation_user_id=None):
    global access_token
    # Authenticate using API key if access_token is None
    if access_token is None:
        success = login_with_api(user, api_key)
        if not success:
            print("Authentication failed.")
            return None
    url = f"{API_BASE_URL}/flow-stream"
    payload = {
        "flow_id": flow_name,
        "message": message,
        "history": history,
        "vault": vault,
        "conversation_user_id": conversation_user_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    full_response = ''
    logs = []
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            event_type = None  # Initialize event_type
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('event: '):
                        event_type = decoded_line[7:].strip()
                    elif decoded_line.startswith('data: '):
                        data = json.loads(decoded_line[6:])
                        if event_type == 'message':
                            content = data.get('content', '')
                            full_response += content
                            print(content, end='', flush=True)
                        elif event_type == 'log':
                            logs.append(data)
                        elif event_type == 'done':
                            break
            return {"response": full_response, "logs": logs}
    except requests.exceptions.RequestException as e:
        print(f"Error during run_flow_stream: {e}")
        return None
