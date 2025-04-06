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
from .credentials_manager import CredentialsManager
from .cloud_api import call_proj, call_req
from .itemize import cloud_name

class CloudManager:
    def __init__(self, user: str, api_key: str, vault: str):
        self.user = user
        self.api = api_key
        self.vault = vault
        # Create credentials
        self.credentials = CredentialsManager(user, self.api)
        # Instantiate the client 
        self.storage_client = storage.Client(project=call_proj(), credentials=self.credentials)
        self.username = self.username(self.user)
        self.cloud = self.storage_client.bucket(self.username)
        self.cloud_name = cloud_name
        self.req_count = 0 

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
        blob = self.cloud.blob('/vaults_list')
        return json.loads(blob.download_as_text())

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
        # Create a temporary file with the desired extension
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(suffix='.json')
        try:
            blob = self.cloud.blob(vault_name)
            blob.download_to_filename(temp_file_path)
        finally:
            # Close the file descriptor
            os.close(temp_file_descriptor)
        return temp_file_path
    
    def upload(self, item, text, meta, vault = None):
        vault if vault else self.vault
        self.upload_to_cloud(self.cloud_name(vault, item, self.user, self.api, item=True), text)
        self.upload_to_cloud(self.cloud_name(vault, item, self.user, self.api, meta=True), json.dumps(meta))
        
    def upload_vaults_list(self, vaults_list):
        blob = self.cloud.blob('/vaults_list')
        blob.upload_from_string(json.dumps(vaults_list))
        
    def upload_personality_message(self, personality_message):
        self.upload_to_cloud(f'{self.vault}/personality_message', personality_message)
    
    def upload_custom_prompt(self, prompt):
        self.upload_to_cloud(f'{self.vault}/prompt', prompt)
     
    def username(self, input_string):
        return input_string.replace("@", "_at_").replace(".", "_dot_") + '_vvclient'

    def get_mapping(self):
        temp_file_path = self.download_to_temp_file(f'{self.username}.json')
        with open(temp_file_path, 'r') as json_file:
            _map = json.load(json_file)
        os.remove(temp_file_path)
        return _map
    
    def build_update(self, n):
        _map = self.get_mapping()
        for i in range(len(_map)):
            if _map[i]['vault'] == self.vault:
                _map[i]['last_use'] = time.time()
                try:
                    _map[i]['total_use'] += 1
                except:
                    _map[i]['total_use'] = 1

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            _path = temp_file.name
            json.dump(_map, temp_file, indent=2)
            call_req(self.user, self.api, n)
            
        self.upload_temp_file(_path, f'{self.username}.json')
    
    def build_data_update(self):
        _map = self.get_mapping()
        for i in range(len(_map)):
            if _map[i]['vault'] == self.vault:
                _map[i]['last_update'] = time.time()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            _path = temp_file.name
            json.dump(_map, temp_file, indent=2)
            
        self.upload_temp_file(_path, f'{self.username}.json')
    
    def delete_blob(self, blob):
        blob.delete()

    def delete(self):
        blobs = self.cloud.list_blobs(prefix=self.vault)
        # Delete each object concurrently
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
        # Option 1: Just upload an empty file to mark it as a directory
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
        
        # Ensure the base path ends with a slash for proper prefix matching
        if not base_path.endswith('/'):
            base_path += '/'
        
        # Get all objects with this prefix
        all_objects = self.cloud_manager.list_objects(base_path)
        
        # Process results to find immediate children
        result = []  # Use a list instead of a set
        directories = set()  # Set is fine for strings
        
        for obj_path in all_objects:
            # Skip the base path itself
            if obj_path == base_path:
                continue
                
            # Remove the base path to get the relative path
            rel_path = obj_path[len(base_path):]
            
            # If it contains a slash, it's in a subdirectory
            if '/' in rel_path:
                # The first segment is a directory
                dir_name = rel_path.split('/')[0]
                directories.add(dir_name)
            else:
                # It's an immediate item in this directory
                result.append({"name": rel_path, "type": "item"})
        
        # Add all identified directories
        for dir_name in directories:
            # Check if this directory is already in the result
            if not any(item.get("name") == dir_name and item.get("type") == "directory" for item in result):
                result.append({"name": dir_name, "type": "directory"})
        
        # Sort the result
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
        
        # Check if it's an item or directory by listing objects
        objects_to_delete = self.cloud_manager.list_objects(base_path)

        if not objects_to_delete:
            # Possibly no object, means maybe it's a single file
            # Attempt single-file delete
            blob = self.cloud_manager.cloud.blob(base_path)
            if blob.exists(self.cloud_manager.storage_client):
                self.cloud_manager.delete_blob(blob)
        else:
            # If we have multiple objects, it's definitely a directory
            for obj_path in objects_to_delete:
                blob = self.cloud_manager.cloud.blob(obj_path)
                if blob.exists(self.cloud_manager.storage_client):
                    self.cloud_manager.delete_blob(blob)
            
            # Also delete the directory marker
            marker_blob = self.cloud_manager.cloud.blob(base_path + "/.directory")
            if marker_blob.exists(self.cloud_manager.storage_client):
                self.cloud_manager.delete_blob(marker_blob)