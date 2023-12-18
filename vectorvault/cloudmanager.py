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
        blobs = self.cloud.list_blobs(prefix=f'{vault}')
        directories = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) == 2:
                vault_name = parts[1]
                if vault_name.endswith('.ann'):
                    clean_vault_name = vault_name.replace('.ann', '')
                    directories.add(clean_vault_name)
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

    def upload(self, item, text, meta):
        self.upload_to_cloud(self.cloud_name(self.vault, item, self.user, self.api, item=True), text)
        self.upload_to_cloud(self.cloud_name(self.vault, item, self.user, self.api, meta=True), json.dumps(meta))
    
    def upload_personality_message(self, personality_message):
        self.upload_to_cloud(f'{self.vault}/personality_message', personality_message)
    
    def upload_custom_prompt(self, prompt):
        self.upload_to_cloud(f'{self.vault}/prompt', prompt)
    
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
