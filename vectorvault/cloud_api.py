import requests
import hashlib
import json
import time
from requests.exceptions import JSONDecodeError


class TokenManager:
    """Manages access and refresh tokens for a specific user instance"""
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
    
    def set_tokens(self, access_token, refresh_token=None):
        self.access_token = access_token
        if refresh_token is not None:
            self.refresh_token = refresh_token
    
    def clear_tokens(self):
        self.access_token = None
        self.refresh_token = None
    
    def has_access_token(self):
        return self.access_token is not None


class CloudAPI:
    """Manages all cloud API calls with instance-specific state"""
    
    def __init__(self, user, api_key):
        self.user = user
        self.api = api_key
        self.token_manager = TokenManager()
        self.API_BASE_URL = "https://api.vectorvault.io" 
        self.VECTOR_BASE_URL = "https://vectors.vectorvault.io"
    
    def request_name_vecs(self, vault, bytesize=None):
        url = f'{self.VECTOR_BASE_URL}/name_vecs'
        headers = {'Content-Type': 'application/json'}
        data = {
            "vault": vault,
            "user": self.user,
            "api_key": self.api
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

    def call_name_vecs(self, vault, bytesize=None):
        url = f'{self.VECTOR_BASE_URL}/name_vecs'
        headers = {'Content-Type': 'application/json'}
        data = {
            "vault": vault,
            "user": self.user,
            "api_key": self.api
        }
        if bytesize and url and headers:
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

    @staticmethod
    def _static_call_name_vecs(user, api, vault, bytesize=None):
        url = f'https://vectors.vectorvault.io/name_vecs'
        headers = {'Content-Type': 'application/json'}
        data = {
            "vault": vault,
            "user": user,
            "api_key": api
        }
        if bytesize and url and headers:
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

    def call_buildpath(self, v, x, bytesize=None):
        data = {
            "v": v,
            "x": x,
            "user": self.user,
            "api_key": self.api,
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
    
    @staticmethod
    def _static_call_buildpath(user, api, v, x, bytesize=None):
        data = {
            "v": v,
            "x": x,
            "user": user,
            "api_key": api,
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
    
    def call_req(self, num=4):
        url = f"{self.API_BASE_URL}/make_request"
        payload = {
            'user': self.user,
            'api_key': self.api,
            'num': num,
        }
        response = requests.post(url, json=payload)
        
        try:
            return response.json()
        except JSONDecodeError:
            return None

    @staticmethod
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

    def login_with_api(self):
        url = f"{self.API_BASE_URL}/login_with_api"
        payload = {
            "email": self.user,
            "api_key": self.api
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
            self.token_manager.set_tokens(access_token, refresh_token)
            return True
        except requests.exceptions.RequestException as e:
            return False

    def refresh_access_token(self, max_retries=1):
        """
        Attempts to refresh the access token. Falls back quickly to fresh login on failure.
        """
        if not self.token_manager.refresh_token:

            self.token_manager.clear_tokens()
            return self.login_with_api()
        
        url = f"{self.API_BASE_URL}/refresh"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token_manager.refresh_token}"
        }
        
        try:
            response = requests.post(url, headers=headers, timeout=5)
            
            if response.status_code == 404:
                self.token_manager.clear_tokens()
                return self.login_with_api()
            
            if response.status_code != 200:
                self.token_manager.clear_tokens()
                return self.login_with_api()
                
            data = response.json()
            

            access_token = data.get('access_token')
            refresh_token = data.get('refresh_token', self.token_manager.refresh_token)
            self.token_manager.set_tokens(access_token, refresh_token)
            return True
            
        except requests.exceptions.RequestException as e:

            self.token_manager.clear_tokens()
            return self.login_with_api()

    def call_cloud_save(self, vault, embeddings_model, text, meta=None, name=None, split=None, split_size=None):
        current_token = self.get_access_token()
        url = f"{self.API_BASE_URL}/add_cloud"
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
            "Authorization": f"Bearer {current_token}"
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401 or e.response.status_code == 404:

                if self.refresh_access_token():
                    headers["Authorization"] = f"Bearer {self.token_manager.access_token}"
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

    def run_flow(self, flow_name, message, history='', invoke_method=None, 
                internal_vars=None, image_url=None, max_retries=5, **kwargs):
        
        for attempt in range(max_retries + 1):  

            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.API_BASE_URL}/flow"
            payload = {
                "flow_id": flow_name,
                "message": message,
                "history": history,
                'invoke_method': invoke_method, 
                'internal_vars': internal_vars, 
                'image_url': image_url
            }

            payload.update(kwargs)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_token}"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 404:
                    if not self.refresh_access_token():
                        return None
                    if attempt >= max_retries:
                        return None
                    continue
                
                response.raise_for_status()
                result = response.json()
                return result.get('response', None)
                            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    if not self.refresh_access_token():
                        return None
                    if attempt >= max_retries:
                        return None
                    continue
                else:
                    return None
                    
            except requests.exceptions.RequestException as e:
                return None

    def run_flow_stream(self, flow_name, message, history='', invoke_method=None, 
                       internal_vars=None, image_url=None, max_retries=5, **kwargs):
        
        for attempt in range(max_retries + 1): 
            try:
                current_token = self.get_access_token()
                if not current_token:
                    yield "error: Failed to obtain access token."
                    return
                    
                url = f"{self.API_BASE_URL}/flow-stream"
                payload = {
                    "flow_id": flow_name,
                    "message": message,
                    "history": history,
                    'invoke_method': invoke_method, 
                    'internal_vars': internal_vars, 
                    'image_url': image_url
                }

                payload.update(kwargs)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {current_token}"
                }
                
                with requests.post(url, json=payload, headers=headers, stream=True) as response:

                    if response.status_code == 404:
                        if not self.refresh_access_token():
                            yield "error: Failed to refresh token, cannot continue."
                            return
                        if attempt >= max_retries:
                            yield "error: Endpoint not found after multiple attempts."
                            return
                        continue
                    
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            yield decoded_line
                            

                    return
                                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    if not self.refresh_access_token():
                        yield "error: Failed to refresh token, cannot continue."
                        return
                    if attempt >= max_retries:
                        yield "error: Authorization failed after multiple attempts."
                        return
                    continue
                else:
                    yield f"error: {str(e)}"
                    return
                    
            except requests.exceptions.RequestException as e:
                yield f"error: {str(e)}"
                return

    def get_access_token(self):
        """Get or create access token for this instance"""
        if not self.token_manager.has_access_token():
            success = self.login_with_api()
            if not success:
                return None
        return self.token_manager.access_token

    def get_init_data(self, vault, max_retries=1):
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
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/init-data/{vault}"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=2)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return None
        
        except requests.exceptions.RequestException:
            return None

    def get_vault_metadata(self, max_retries=3):
        """
        Get all vault metadata for the user from backend API.
        Returns list of vault metadata dicts.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/vault-metadata"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def update_vault_metadata(self, vault, total_items=None, last_update=None, last_use=None, total_use=None, max_retries=3):
        """
        Update vault metadata on backend API.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/vault-metadata/update"
            
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
                    if not self.refresh_access_token():
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

    def increment_vault_usage(self, vault, n=4, max_retries=3):
        """
        Increment vault usage counter on backend API.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/vault-metadata/increment-usage"
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
                    if not self.refresh_access_token():
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

    def get_vault_mapping(self, vault, max_retries=3):
        """
        Get item mapping for a vault from backend API.
        Returns dict mapping item_id -> UUID.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/vault-mapping/{vault}"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def update_vault_mapping(self, vault, mapping, max_retries=3):
        """
        Update item mapping for a vault on backend API.
        """
        if not isinstance(mapping, dict):
            mapping = {}
        
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/vault-mapping/{vault}"
            payload = {"item_mapping": mapping}
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def get_vaults_list(self, max_retries=3):
        """
        Get list of all vaults from backend API.
        Returns list of vault names.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/vaults-list"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=2)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
                        return None
                    if attempt >= max_retries:
                        return None
                    time.sleep(0.5)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                vaults = result.get('vaults', [])
                if isinstance(vaults, dict):
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

    def update_vaults_list(self, vaults, max_retries=3):
        """
        Update list of all vaults on backend API.
        Ensures vaults is always sent as a list.
        """
        if isinstance(vaults, dict):
            vault_list = []
            for key, value in vaults.items():
                if key.isdigit():
                    vault_list.append(value)
            vaults = sorted(vault_list)
        elif not isinstance(vaults, list):
            vaults = []
        
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/vaults-list"
            payload = {"vaults": vaults}
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def get_custom_prompt(self, vault, context=True, max_retries=3):
        """
        Get custom prompt for a vault from backend API.
        Returns prompt text or None if doesn't exist.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/fetch_custom_prompt"
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
                    if not self.refresh_access_token():
                        return None
                    if attempt >= max_retries:
                        return None
                    time.sleep(0.5)
                    continue
                
                response.raise_for_status()
                result = response.json()
                return result if result else None
                            
            except requests.exceptions.RequestException as e:
                if attempt >= max_retries:
                    return None
                time.sleep(0.5)
                continue
        
        return None

    def save_custom_prompt(self, vault, prompt, context=True, max_retries=3):
        """
        Save custom prompt for a vault on backend API.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/save_custom_prompt"
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
                    if not self.refresh_access_token():
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

    def get_personality_message(self, vault, max_retries=3):
        """
        Get personality message for a vault from backend API.
        Returns personality message text or None if doesn't exist.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/fetch_personality_message"
            payload = {"vault": vault}
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=2)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def save_personality_message(self, vault, personality_message, max_retries=3):
        """
        Save personality message for a vault on backend API.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/save_personality_message"
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
                    if not self.refresh_access_token():
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

    def get_user_vault_data(self, max_retries=3):
        """
        Get the master vault data file for the user.
        Returns list of all vaults with their metadata.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return None
                
            url = f"{self.VECTOR_BASE_URL}/user-vault-data"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def update_user_vault_data(self, vault_data, max_retries=3):
        """
        Update the master vault data file for the user.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/user-vault-data"
            payload = {"vault_data": vault_data}
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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

    def delete_vault_metadata(self, vault, max_retries=3):
        """
        Delete vault metadata from backend API.
        """
        for attempt in range(max_retries + 1):
            current_token = self.get_access_token()
            if not current_token:
                return False
                
            url = f"{self.VECTOR_BASE_URL}/vault-metadata/{vault}"
            headers = {
                "Authorization": f"Bearer {current_token}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.delete(url, headers=headers, timeout=10)
                
                if response.status_code == 404 or response.status_code == 401:
                    if not self.refresh_access_token():
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