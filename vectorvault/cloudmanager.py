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
from .creds import CustomCredentials
from .vecreq import call_proj
from .itemize import cloud_name
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

class CloudManager:
    def __init__(self, user: str, api_key: str, vault: str):
        self.user = user
        self.api = api_key
        self.vault = vault
        # Creates the credentials
        self.credentials = CustomCredentials(user, self.api)
        # Instantiates the client 
        self.storage_client = storage.Client(project=call_proj(), credentials=self.credentials)
        self.cloud = self.storage_client.bucket(self.get_bkt(self.user))
        self.cloud_name = cloud_name

    def vault_exists(self, vault_name):
        return storage.Blob(bucket=self.cloud, name=vault_name).exists(self.storage_client)
    
    def list_vaults(self, vault):
        blobs = self.gcloud.list_blobs(prefix=f'{vault}')
        directories = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) > 2 and not parts[1][0].isdigit():
                directories.add(parts[1])
        return list(directories)
    
    def upload_to_cloud(self, vault_name, content):
        blob = self.cloud.blob(vault_name)
        blob.upload_from_string(content)

    def download_text_from_cloud(self, vault_name):
        blob = self.cloud.blob(vault_name)
        return blob.download_as_text()

    def upload_temp_file(self, temp_file_path, vault_name):
        blob = self.cloud.blob(vault_name)
        blob.upload_from_filename(temp_file_path)
        os.remove(temp_file_path)

    def download_to_temp_file(self, vault_name):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob = self.cloud.blob(vault_name)
            blob.download_to_filename(temp_file.name) 
            return temp_file.name

    def upload(self, item, text, meta):
        self.upload_to_cloud(self.cloud_name(self.vault, item, self.user, self.api, item=True), text)
        self.upload_to_cloud(self.cloud_name(self.vault, item, self.user, self.api, meta=True), json.dumps(meta))
    
    def delete_blob(self, blob):
        blob.delete()
     
    def get_bkt(self, input_string):
        return input_string.replace("@", "_at_").replace(".", "_dot_") + '_vvclient'

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
    
    def rename_item(self, old_item : int, new_item : int):
        old_path = self.cloud_name(self.vault, old_item, self.user, self.api, item=True)
        new_path = self.cloud_name(self.vault, new_item, self.user, self.api, item=True)
        old_meta_path = self.cloud_name(self.vault, old_item, self.user, self.api, meta=True)
        new_meta_path = self.cloud_name(self.vault, new_item, self.user, self.api, meta=True)
        
        old_blob = self.cloud.blob(old_path)
        new_blob = self.cloud.blob(new_path)
        old_meta_blob = self.cloud.blob(old_meta_path)
        new_meta_blob = self.cloud.blob(new_meta_path)
        
        if old_blob.exists(self.storage_client):
            self.cloud.copy_blob(old_blob, self.cloud, new_blob.name)
            old_blob.delete()
        else:
            print(f"Item at path {old_path} does not exist.")
        if old_meta_blob.exists(self.storage_client):
            meta_data_string = old_meta_blob.download_as_string()
            meta_data = json.loads(meta_data_string)
            meta_data['item_id'] -= 1
            new_meta_blob.upload_from_string(json.dumps(meta_data))
            old_meta_blob.delete()
        else:
            print(f"Item at path {old_path} does not exist.")
            
    def delete_and_rename_all_items_after(self, item_id):
        # Delete the specified item
        self.delete_item(item_id)
        
        # Start renaming subsequent items
        current_item_id = item_id + 1
        while True:
            next_item_id = current_item_id - 1
            
            # Check if the current item exists
            current_item_path = self.cloud_name(self.vault, current_item_id, self.user, self.api, item=True)
            current_blob = self.cloud.blob(current_item_path)
            
            if current_blob.exists(self.storage_client):
                # Rename the current item to next_item_id
                self.rename_item(current_item_id, next_item_id)
                current_item_id += 1
            else:
                # If the blob doesn't exist, break out of the loop
                break