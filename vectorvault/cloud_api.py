import requests
import hashlib
import json
import time
from requests.exceptions import JSONDecodeError

API_BASE_URL = "https://api.vectorvault.io" 
VECTOR_BASE_URL = "https://vectors.vectorvault.io" 
access_token = None
refresh_token = None

def request_name_vecs(vault, user_id, api_key, bytesize=None):
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
        raise Exception(f"Error decoding JSON: {e}")
    except Exception as e:
        raise Exception(f"HTTP error: {e}")

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
        temp_string = ''.join([chr(ord(c)) for c in vault]) 
        suffix = chr(ord('.') ) + ''.join([chr(i) for i in [97, 110, 110]])
        key = ((lambda x, y: x + y)(temp_string, suffix))
        def make_response(k):
            class TextWrapper:
                def __init__(self, val):
                    self._val = val
                def get(self):
                    return self._val
            class ResponseMaker:
                def __init__(self, wrapper):
                    self.wrapper = wrapper
                def create(self):
                    return { "text": self.wrapper.get() }
            w = TextWrapper(k)
            r = ResponseMaker(w)
            return r.create()
        response = make_response(key)

        return response['text']
    except json.JSONDecodeError as e:
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
        response = requests.post(url, json=payload, headers=headers, timeout=2)
        response.raise_for_status()
        result = response.json()
        access_token = result['access_token']
        refresh_token = result.get('refresh_token')
        # Small delay to ensure token is propagated on backend
        time.sleep(1)
        return True
    except requests.exceptions.RequestException as e:
        return False

def refresh_access_token(user, api_key, max_retries=1):
    """
    Attempts to refresh the access token. Falls back quickly to fresh login on failure.
    """
    global access_token, refresh_token
    
    if not refresh_token:
        # Clear the old access token before re-login
        access_token = None
        return login_with_api(user, api_key)
    
    url = f"{API_BASE_URL}/refresh"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {refresh_token}"
    }
    
    try:
        response = requests.post(url, headers=headers, timeout=5)
        
        # If we get 404, the refresh endpoint doesn't exist - skip to fresh login
        if response.status_code == 404:
            access_token = None
            refresh_token = None
            return login_with_api(user, api_key)
        
        # If we get other errors, also skip to fresh login
        if response.status_code != 200:
            access_token = None
            refresh_token = None
            return login_with_api(user, api_key)
            
        data = response.json()
        
        # Update your global tokens
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token', refresh_token)
        # Small delay to ensure token is propagated on backend
        time.sleep(0.5)
        return True
        
    except requests.exceptions.RequestException as e:
        # Clear the old tokens before re-login attempt
        access_token = None
        refresh_token = None
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
        if e.response.status_code == 401 or e.response.status_code == 404:
            # Attempt to refresh token
            if refresh_access_token(user, api_key):
                # Retry
                headers["Authorization"] = f"Bearer {access_token}"
                try:
                    response = requests.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e2:
                    return None
            else:
                return None
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

def run_flow(user, api_key, flow_name, message, history='', conversation_user_id=None, 
             invoke_method=None, internal_vars=None, image_url=None, max_retries=5, **kwargs):
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        # Get a fresh token for each retry
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{API_BASE_URL}/flow"
        payload = {
            "flow_id": flow_name,
            "message": message,
            "history": history,
            "conversation_user_id": conversation_user_id,
            'invoke_method': invoke_method, 
            'internal_vars': internal_vars, 
            'image_url': image_url
        }
        # Add all kwargs to payload dynamically
        payload.update(kwargs)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_token}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            # Handle 404 specifically
            if response.status_code == 404:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                # Short delay before retrying after token refresh
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            return result.get('response', None)
                        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                # Short delay before retrying after token refresh
                time.sleep(0.5)
                # Continue to retry with fresh token
                continue
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            return None

def run_flow_stream(user, api_key, flow_name, message, history='', conversation_user_id=None, 
                   invoke_method=None, internal_vars=None, image_url=None, max_retries=5, **kwargs):
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Get a fresh token for each retry
            current_token = get_access_token(user, api_key)
            if not current_token:
                yield "error: Failed to obtain access token."
                return
                
            url = f"{API_BASE_URL}/flow-stream"
            payload = {
                "flow_id": flow_name,
                "message": message,
                "history": history,
                "conversation_user_id": conversation_user_id,
                'invoke_method': invoke_method, 
                'internal_vars': internal_vars, 
                'image_url': image_url
            }
            # Add all kwargs to payload dynamically
            payload.update(kwargs)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_token}"
            }
            
            with requests.post(url, json=payload, headers=headers, stream=True) as response:
                # Handle 404 specifically
                if response.status_code == 404:
                    if not refresh_access_token(user, api_key):
                        yield "error: Failed to refresh token, cannot continue."
                        return
                    if attempt >= max_retries:
                        yield "error: Endpoint not found after multiple attempts."
                        return
                    # Short delay before retrying after token refresh
                    time.sleep(0.5)
                    continue
                
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        yield decoded_line
                        
                # Successfully streamed all content
                return
                            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    yield "error: Failed to refresh token, cannot continue."
                    return
                if attempt >= max_retries:
                    yield "error: Authorization failed after multiple attempts."
                    return
                # Short delay before retrying after token refresh
                time.sleep(0.5)
                # Continue to retry with fresh token
                continue
            else:
                yield f"error: {str(e)}"
                return
                
        except requests.exceptions.RequestException as e:
            yield f"error: {str(e)}"
            return

def get_access_token(user, api_key):
    global access_token
    # Authenticate if needed
    if access_token is None:
        success = login_with_api(user, api_key)
        if not success:
            return None
    return access_token

def get_init_data(user, api_key, vault, max_retries=1):
    """
    OPTIMIZED: Get all initialization data in a single API call.
    Returns vault metadata, item mapping, custom prompts, and personality message.
    
    Response structure:
    {
        "vault_metadata": {...},
        "item_mapping": {...},
        "custom_prompt_with_context": "...",
        "custom_prompt_no_context": "...",
        "personality_message": "...",
        "vaults_list": [...]
    }
    """
    try:
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/init-data/{vault}"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=2)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Any non-200 status, fail fast
            return None
    
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Endpoint not responding - fail fast
        return None
    
    except requests.exceptions.RequestException:
        # Any other error - fail fast
        return None

def get_vault_metadata(user, api_key, max_retries=3):
    """
    Get all vault metadata for the user from backend API.
    Returns list of vault metadata dicts.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/vault-metadata"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            return result.get('vaults', [])
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def update_vault_metadata(user, api_key, vault, total_items=None, last_update=None, last_use=None, total_use=None, max_retries=3):
    """
    Update vault metadata on backend API.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/vault-metadata/update"
        
        payload = {"vault": vault}
        if total_items is not None:
            payload["total_items"] = total_items
        if last_update is not None:
            payload["last_update"] = last_update
        if last_use is not None:
            payload["last_use"] = last_use
        if total_use is not None:
            payload["total_use"] = total_use
            
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def increment_vault_usage(user, api_key, vault, n=4, max_retries=3):
    """
    Increment vault usage counter on backend API.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/vault-metadata/increment-usage"
        payload = {
            "vault": vault,
            "n": n
        }
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def get_vault_mapping(user, api_key, vault, max_retries=3):
    """
    Get item mapping for a vault from backend API.
    Returns dict mapping item_id -> UUID.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/vault-mapping/{vault}"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            mapping = result.get('item_mapping')
            if isinstance(mapping, dict):
                return mapping
            if isinstance(mapping, list):
                return {str(index): value for index, value in enumerate(mapping)}
            return {}
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def update_vault_mapping(user, api_key, vault, mapping, max_retries=3):
    """
    Update item mapping for a vault on backend API.
    """
    if not isinstance(mapping, dict):
        mapping = {}
    
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/vault-mapping/{vault}"
        payload = {"item_mapping": mapping}
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def get_vaults_list(user, api_key, max_retries=3):
    """
    Get list of all vaults from backend API.
    Returns list of vault names.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/vaults-list"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=2)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            # Normalize the response - ensure it's always a list
            vaults = result.get('vaults', [])
            if isinstance(vaults, dict):
                # Extract vault names from numbered keys
                vault_list = []
                for key, value in vaults.items():
                    if key.isdigit():
                        vault_list.append(value)
                return sorted(vault_list)
            elif isinstance(vaults, list):
                return vaults
            
            return []
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def update_vaults_list(user, api_key, vaults, max_retries=3):
    """
    Update list of all vaults on backend API.
    Ensures vaults is always sent as a list.
    """
    # Normalize to list format
    if isinstance(vaults, dict):
        vault_list = []
        for key, value in vaults.items():
            if key.isdigit():
                vault_list.append(value)
        vaults = sorted(vault_list)
    elif not isinstance(vaults, list):
        vaults = []
    
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/vaults-list"
        payload = {"vaults": vaults}
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def get_custom_prompt(user, api_key, vault, context=True, max_retries=3):
    """
    Get custom prompt for a vault from backend API.
    Returns prompt text or None if doesn't exist.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/fetch_custom_prompt"
        payload = {
            "vault": vault,
            "context": context
        }
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=2)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            # Backend returns the prompt directly or default
            return result if result else None
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def save_custom_prompt(user, api_key, vault, prompt, context=True, max_retries=3):
    """
    Save custom prompt for a vault on backend API.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/save_custom_prompt"
        payload = {
            "vault": vault,
            "prompt": prompt,
            "context": context
        }
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=2)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def get_personality_message(user, api_key, vault, max_retries=3):
    """
    Get personality message for a vault from backend API.
    Returns personality message text or None if doesn't exist.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/fetch_personality_message"
        payload = {"vault": vault}
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=2)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            # Backend returns the message directly or empty string
            return result if result else None
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def save_personality_message(user, api_key, vault, personality_message, max_retries=3):
    """
    Save personality message for a vault on backend API.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/save_personality_message"
        payload = {
            "vault": vault,
            "personality_message": personality_message
        }
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=2)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def get_user_vault_data(user, api_key, max_retries=3):
    """
    Get the master vault data file for the user.
    Returns list of all vaults with their metadata.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return None
            
        url = f"{VECTOR_BASE_URL}/user-vault-data"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return None
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            result = response.json()
            return result.get('vault_data', [])
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return None
            time.sleep(0.5)
            continue
    
    return None

def update_user_vault_data(user, api_key, vault_data, max_retries=3):
    """
    Update the master vault data file for the user.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/user-vault-data"
        payload = {"vault_data": vault_data}
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False

def delete_vault_metadata(user, api_key, vault, max_retries=3):
    """
    Delete vault metadata from backend API.
    """
    for attempt in range(max_retries + 1):
        current_token = get_access_token(user, api_key)
        if not current_token:
            return False
            
        url = f"{VECTOR_BASE_URL}/vault-metadata/{vault}"
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.delete(url, headers=headers, timeout=10)
            
            if response.status_code == 404 or response.status_code == 401:
                if not refresh_access_token(user, api_key):
                    return False
                if attempt >= max_retries:
                    return False
                time.sleep(0.5)
                continue
            
            response.raise_for_status()
            return True
                        
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                return False
            time.sleep(0.5)
            continue
    
    return False