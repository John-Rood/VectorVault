# VECTOR VAULT CONFIDENTIAL
# __________________
# 
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Vector Vault and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Vector Vault
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Vector Vault. See license for consent.

import tempfile
import os
import json
import time
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from .itemize import cloud_name, name_map
from .credentials_manager import CredentialsManager
from .cloud_api import CloudAPI

class CloudManager:
    def __init__(self, user: str, api_key: str, vault: str):
        self.user = user
        self.api = api_key
        self.vault = vault
        self.cloud_api = CloudAPI(user, api_key)
        self._storage_client = None
        self._cloud = None
        self._username = None
        self.cloud_name = cloud_name
        self.req_count = 0
        
        self._init_data = None
        self._init_data_loaded = False
        self.custom_prompt_with_context = None
        self.custom_prompt_no_context = None
        self.personality_message = None
        self.item_mapping = {}
        self.vault_metadata = {}
        self.load_init_data()
    
    @property
    def credentials(self):
        """Lazy initialization of credentials"""
        if not hasattr(self, '_credentials'):
            self._credentials = CredentialsManager(self.user, self.api)
        return self._credentials
    
    @property
    def storage_client(self):
        """Lazy initialization of storage client"""
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self.cloud_api.call_proj(), credentials=self.credentials)
        return self._storage_client
    
    @property
    def username(self):
        """Lazy initialization of username"""
        if self._username is None:
            self._username = self.user.replace("@", "_at_").replace(".", "_dot_") + '_vvclient'
        return self._username
    
    @property
    def cloud(self):
        """Lazy initialization of cloud bucket"""
        if self._cloud is None:
            self._cloud = self.storage_client.bucket(self.username)
        return self._cloud
    
    def load_init_data(self):
        """
        OPTIMIZED: Load all initialization data in a single API call.
        Called immediately on CloudManager initialization.
        """
        if self._init_data_loaded:
            return
        
        self._init_data_loaded = True
        result = self.cloud_api.get_init_data(self.vault)
        
        if result is not None:
            self._init_data = result
            self.custom_prompt_with_context = result.get('custom_prompt_with_context')
            self.custom_prompt_no_context = result.get('custom_prompt_no_context')
            self.personality_message = result.get('personality_message')
            self.item_mapping = self._normalize_item_mapping(result.get('item_mapping'))
            self.vault_metadata = self._extract_metadata_for_vault(result.get('vault_metadata'))
        else:
            self._init_data = None 

    def vault_exists(self, vault_name):
        return storage.Blob(bucket=self.cloud, name=vault_name).exists(self.storage_client)
    
    def list_vaults(self, vault):
        blobs = self.cloud.list_blobs(prefix=f'{vault}')
        directories = set()
        for blob in blobs:
            if blob.name.endswith('.ann'):
                if vault:
                    parts = blob.name.split('/')
                    if len(parts) == 2:
                        vault_name = parts[1]
                        if vault_name.endswith('.ann'):
                            clean_vault_name = vault_name.replace('.ann', '')
                            directories.add(clean_vault_name)
                else:
                    vault_name = blob.name.split('/')[0] 
                    clean_vault_name = vault_name.replace('.ann', '')
                    directories.add(clean_vault_name)
        return sorted(list(directories))
    
    def upload_to_cloud(self, vault_name, content):
        blob = self.cloud.blob(vault_name)
        blob.upload_from_string(content)

    def download_vaults_list_from_cloud(self):
        """Get vaults list - always returns a list"""
        if self._init_data and 'vaults_list' in self._init_data:
            vaults_list = self._init_data['vaults_list']
        else:
            vaults_list = self.cloud_api.get_vaults_list()
        
        if isinstance(vaults_list, dict):
            result = []
            for key, value in vaults_list.items():
                if key.isdigit():
                    result.append(value)
            return sorted(result)
        elif isinstance(vaults_list, list):
            return vaults_list
        
        return []

    def download_text_from_cloud(self, vault_name):
        blob = self.cloud.blob(vault_name)
        return blob.download_as_text()

    def upload_temp_file(self, temp_file_path, vault_name):
        blob = self.cloud.blob(vault_name)
        blob.upload_from_filename(temp_file_path)

    def download_to_temp_file(self, vault_name):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob = self.cloud.blob(vault_name)
            blob.download_to_filename(temp_file.name) 
            return temp_file.name
    
    def download_json(self, vault_name):
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(suffix='.json')
        try:
            blob = self.cloud.blob(vault_name)
            blob.download_to_filename(temp_file_path)
        finally:
            os.close(temp_file_descriptor)
        return temp_file_path
    
    def download_old_gcs_mapping(self, vault_name):
        try:
            map_filename = name_map(vault_name, self.user, self.api)
            blob = self.cloud.blob(map_filename)
            
            if not blob.exists(self.storage_client):
                return None
            
            map_content = blob.download_as_text()
            mapping = json.loads(map_content)
            
            return mapping
        except Exception as e:
            return None
    
    def upload(self, item, text, meta, vault = None):
        vault if vault else self.vault
        self.upload_to_cloud(self.cloud_name(vault, item, self.user, self.api, item=True), text)
        self.upload_to_cloud(self.cloud_name(vault, item, self.user, self.api, meta=True), json.dumps(meta))
        
    def upload_vaults_list(self, vaults_list):
        """Update vaults list - ensure it's always sent as a list"""
        if isinstance(vaults_list, dict):
            result = []
            for key, value in vaults_list.items():
                if key.isdigit():
                    result.append(value)
            vaults_list = sorted(result)
        
        self.cloud_api.update_vaults_list(vaults_list)
        
        self._init_data_loaded = False
        self._init_data = None

    def upload_personality_message(self, personality_message):
        """Save personality message and update local cache"""
        self.personality_message = personality_message
        
        self.cloud_api.save_personality_message(self.vault, personality_message)
    
    def upload_custom_prompt(self, prompt, context=False):
        """Save custom prompt and update local cache"""
        if context:
            self.custom_prompt_with_context = prompt
        else:
            self.custom_prompt_no_context = prompt
        
        self.cloud_api.save_custom_prompt(self.vault, prompt, context)
    
    def download_custom_prompt(self, context=True):
        """Get custom prompt"""
        if context and self.custom_prompt_with_context:
            return self.custom_prompt_with_context
        elif not context and self.custom_prompt_no_context:
            return self.custom_prompt_no_context
    
    def download_personality_message(self):
        """Get personality message"""
        if self.personality_message:
            return self.personality_message

    def get_mapping(self):
        """Get item mapping for current vault"""
        if self.item_mapping:
            return self.item_mapping
        
        self.item_mapping = self._normalize_item_mapping(self.cloud_api.get_vault_mapping(self.vault))
        
        if not self.item_mapping or len(self.item_mapping) == 0:
            gcs_mapping = self.download_old_gcs_mapping(self.vault)
            if gcs_mapping and len(gcs_mapping) > 0:
                self.cloud_api.update_vault_mapping(self.vault, gcs_mapping)
                self.item_mapping = gcs_mapping
                self._init_data_loaded = False
                self._init_data = None
        
        return self.item_mapping

    def _normalize_item_mapping(self, mapping):
        if isinstance(mapping, dict):
            return mapping
        if isinstance(mapping, list):
            return {str(index): value for index, value in enumerate(mapping)}
        return {}

    def _extract_metadata_for_vault(self, source):
        """Normalize metadata source to a per-vault dict."""
        if not source:
            return {}
        
        if isinstance(source, dict):
            if any(key in source for key in ('last_update', 'last_use', 'total_items', 'total_use')):
                return source
            if source.get('vault') == self.vault:
                return {
                    k: v for k, v in source.items()
                    if k in ('last_update', 'last_use', 'total_items', 'total_use')
                }
        
        if isinstance(source, list):
            for entry in source:
                if isinstance(entry, dict) and entry.get('vault') == self.vault:
                    return {
                        k: v for k, v in entry.items()
                        if k in ('last_update', 'last_use', 'total_items', 'total_use')
                    }
        
        return {}

    def get_metadata(self):
        """Get metadata dict for the current vault."""
        if self.vault_metadata:
            return self.vault_metadata
        
        if self._init_data and 'vault_metadata' in self._init_data:
            metadata = self._extract_metadata_for_vault(self._init_data['vault_metadata'])
            if metadata:
                self.vault_metadata = metadata
                return self.vault_metadata
        
        metadata = self._extract_metadata_for_vault(self.cloud_api.get_user_vault_data())
        if metadata:
            self.vault_metadata = metadata
            return self.vault_metadata
        
        metadata = self._extract_metadata_for_vault(self.cloud_api.get_vault_metadata())
        if metadata:
            self.vault_metadata = metadata
            return self.vault_metadata
        
        self.vault_metadata = {}
        return self.vault_metadata
    
    def build_update(self, n):
        """Increment vault usage via backend API"""
        metadata = self.get_metadata()
        current_total_use = metadata.get('total_use', 0) + 1
        
        self.cloud_api.update_vault_metadata(
            self.vault, 
            last_use=time.time(),
            total_use=current_total_use
        )

    def build_data_update(self):
        """Update vault last_update timestamp"""
        now = time.time()
        self.cloud_api.update_vault_metadata(self.vault, last_update=now)
        if not self.vault_metadata:
            self.vault_metadata = {}
        self.vault_metadata['last_update'] = now
    
    def delete_blob(self, blob):
        blob.delete()

    def delete(self):
        blobs = self.cloud.list_blobs(prefix=self.vault)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.delete_blob, blob): blob for blob in blobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to delete blob: {e}")
    
    def delete_item(self, item):
        item_path = self.cloud_name(self.vault, item, self.user, self.api, item=True)
        meta_path = self.cloud_name(self.vault, item, self.user, self.api, meta=True)
        blob = self.cloud.blob(item_path)
        if blob.exists(self.storage_client):
            blob.delete()
        else:
            print(f"Item at path {item_path} does not exist.")
        blob = self.cloud.blob(meta_path)
        if blob.exists(self.storage_client):
            blob.delete()
        else:
            print(f"Item metadata at path {meta_path} does not exist.")

    def item_exists(self, uuid):
        item_path = self.cloud_name(self.vault, uuid, self.user, self.api, item=True)
        blob = self.cloud.blob(item_path)
        return blob.exists(self.storage_client)
        
    def upload_text_to_cloud(self, path, text):
        self.upload_to_cloud(path, text)

    def list_objects(self, prefix):
        blobs = self.cloud.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]


class VaultStorageManager:
    """
    A manager class to handle creation and retrieval of directories and items in the vault.
    """

    def __init__(self, vault: str, cloud_manager):
        self.vault = vault + '/storage'
        self.cloud_manager = cloud_manager

    def create_directory(self, path: str) -> None:
        """
        Creates an empty 'directory marker' or a way to denote a folder at 'path'.
        Implementation depends on how your cloud_manager represents directories.
        For example, you might create a special marker file named ".directory" to
        indicate that this path is a folder.
        """
        directory_marker_path = f"{self.vault}/{path}/.directory"
        self.cloud_manager.upload_text_to_cloud(directory_marker_path, "")
    
    def create_item(self, path: str, value: str) -> None:
        """
        Creates an item at the given path with the provided value.
        If the path already exists as a directory, raise an error or handle accordingly.
        """
        item_path = f"{self.vault}/{path}"
        self.cloud_manager.upload_text_to_cloud(item_path, value)
    
    def list_labels(self, path: str = None) -> list[dict]:
        """
        Lists all items and sub-directories under the given path.
        Automatically detects directories based on common prefixes.
        """
        base_path = f"{self.vault}"
        if path:
            base_path = f"{base_path}/{path}"
        
        if not base_path.endswith('/'):
            base_path += '/'
        
        all_objects = self.cloud_manager.list_objects(base_path)
        result = []
        directories = set()
        
        for obj_path in all_objects:
            if obj_path == base_path:
                continue
                
            rel_path = obj_path[len(base_path):]
            
            if '/' in rel_path:
                dir_name = rel_path.split('/')[0]
                directories.add(dir_name)
            else:
                result.append({"name": rel_path, "type": "item"})
        
        for dir_name in directories:
            if not any(item.get("name") == dir_name and item.get("type") == "directory" for item in result):
                result.append({"name": dir_name, "type": "directory"})
        
        return sorted(result, key=lambda x: x["name"])

    def get_item(self, path: str) -> str:
        """
        Retrieves item text from the path. If the path is a directory, raise an error or return None.
        """
        item_path = f"{self.vault}/{path}"
        value = self.cloud_manager.download_text_from_cloud(item_path)
        return value

    def update_item(self, path: str, new_value: str) -> None:
        """
        Overwrites existing item content. If path is a directory, handle error or do nothing.
        """
        item_path = f"{self.vault}/{path}"
        self.cloud_manager.upload_text_to_cloud(item_path, new_value)

    def delete_label(self, path: str) -> None:
        """
        Deletes a directory (recursively) or a single item, depending on what's at path.
        If there's a .directory marker, remove everything under it. If it's a single file, remove just that file.
        """
        base_path = f"{self.vault}/{path}"
        
        objects_to_delete = self.cloud_manager.list_objects(base_path)

        if not objects_to_delete:
            blob = self.cloud_manager.cloud.blob(base_path)
            if blob.exists(self.cloud_manager.storage_client):
                self.cloud_manager.delete_blob(blob)
        else:
            for obj_path in objects_to_delete:
                blob = self.cloud_manager.cloud.blob(obj_path)
                if blob.exists(self.cloud_manager.storage_client):
                    self.cloud_manager.delete_blob(blob)
            
            marker_blob = self.cloud_manager.cloud.blob(base_path + "/.directory")
            if marker_blob.exists(self.cloud_manager.storage_client):
                self.cloud_manager.delete_blob(marker_blob)