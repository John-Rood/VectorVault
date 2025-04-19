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


def refresh_access_token(user, api_key):
    """
    Attempts to refresh the access token using your API's refresh endpoint.
    Fallback: re-login if refresh_token doesn't work or is None.
    """
    global access_token, refresh_token
    if not refresh_token:
        print("No refresh_token available; re-logging in...")
        return login_with_api(user, api_key)

    url = f"{API_BASE_URL}/refresh_token"
    payload = {"refresh_token": refresh_token}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Update your global tokens
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token')
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error refreshing token: {e}")
        print("Re-logging in as fallback...")
        return login_with_api(user, api_key)


def call_cloud_save(user, api_key, vault, embeddings_model, text, meta=None, name=None, split=None, split_size=None):
    access_token = get_access_token(user, api_key)

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
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            # Attempt to refresh token
            if refresh_access_token(user, api_key):
                # Retry
                headers["Authorization"] = f"Bearer {access_token}"
                try:
                    response = requests.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e2:
                    print(f"Error after refreshing token: {e2}")
                    return None
            else:
                print("Failed to refresh token or re-login.")
                return None
        else:
            print(f"Error during call_cloud_save: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during call_cloud_save: {e}")
        return None


def run_flow(user, api_key, flow_name, message, history='', conversation_user_id=None, 
             parent_save_state_id=None, run_flow_var_name=None, session_id = None, invoke_method = None,
             internal_vars = None):
    access_token = get_access_token(user, api_key)

    url = f"{API_BASE_URL}/flow"
    payload = {
        "flow_id": flow_name,
        "message": message,
        "history": history,
        "conversation_user_id": conversation_user_id,
        'parent_save_state_id': parent_save_state_id, 
        'run_flow_var_name': run_flow_var_name, 
        'session_id': session_id, 
        'invoke_method': invoke_method, 
        'internal_vars': internal_vars, 
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
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            # Attempt token refresh
            if refresh_access_token(user, api_key):
                # Retry
                headers["Authorization"] = f"Bearer {access_token}"
                try:
                    response = requests.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    return result.get('response', None)
                except requests.exceptions.RequestException as e2:
                    print(f"Error after refreshing token: {e2}")
                    return None
            else:
                print("Failed to refresh token or re-login.")
                return None
        else:
            print(f"Error during run_flow: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during run_flow: {e}")
        return None


def run_flow_stream(user, api_key, flow_name, message, history='', conversation_user_id=None, 
                   parent_save_state_id=None, run_flow_var_name=None, session_id=None, invoke_method = None,
                   internal_vars = None):
    access_token = get_access_token(user, api_key)

    url = f"{API_BASE_URL}/flow-stream"
    payload = {
        "flow_id": flow_name,
        "message": message,
        "history": history,
        "conversation_user_id": conversation_user_id,
        'parent_save_state_id': parent_save_state_id, 
        'run_flow_var_name': run_flow_var_name, 
        'session_id': session_id, 
        'invoke_method': invoke_method, 
        'internal_vars': internal_vars, 
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    yield decoded_line
                            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            # Attempt token refresh
            if refresh_access_token(user, api_key):
                # Retry streaming request with new token
                headers["Authorization"] = f"Bearer {access_token}"
                
                try:
                    with requests.post(url, json=payload, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                yield decoded_line
                                        
                except requests.exceptions.RequestException as e2:
                    print(f"Error during second run_flow_stream attempt: {e2}")
                    yield f"error: {str(e2)}"
            else:
                print("Failed to refresh token or re-login.")
                yield "error: Failed to refresh token or re-login."
        else:
            print(f"Error during run_flow_stream: {e}")
            yield f"error: {str(e)}"
    except requests.exceptions.RequestException as e:
        print(f"Error during run_flow_stream: {e}")
        yield f"error: {str(e)}"

def get_access_token(user, api_key):
    global access_token
    # Authenticate if needed
    if access_token is None:
        success = login_with_api(user, api_key)
        if not success:
            print("Authentication failed.")
            return None
    return access_token