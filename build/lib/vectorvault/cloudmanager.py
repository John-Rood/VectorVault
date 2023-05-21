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
# from Vector Vault.

import tempfile
import os
import json
from creds import CustomCredentials
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed


class CloudManager:
    def __init__(self, user: str, api_key: str, vault: str):
        self.user = user
        self.api = api_key
        self.vault = vault

        # Create the OAuth credentials
        credentials = CustomCredentials(self.user, self.api)

        # Instantiates a client with the OAuth2 credentials
        self.storage_client = storage.Client(project='vectorvault-361ab', credentials=credentials)
        self.gcloud = self.storage_client.bucket(self.user)
        print(f'Connected to Vault: {self.vault}')

    def vault_exists(self, vault_name):
        return storage.Blob(bucket=self.gcloud, name=vault_name).exists(self.storage_client)

    def upload_to_cloud(self, vault_name, content):
        blob = self.gcloud.blob(vault_name)
        blob.upload_from_string(content)

    def download_text_from_cloud(self, vault_name):
        blob = self.gcloud.blob(vault_name)
        return blob.download_as_text()

    def upload_temp_file(self, temp_file_path, vault_name):
        blob = self.gcloud.blob(vault_name)
        blob.upload_from_filename(temp_file_path)
        os.remove(temp_file_path)

    def download_to_temp_file(self, vault_name):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob = self.gcloud.blob(vault_name)
            blob.download_to_filename(temp_file.name) 
            return temp_file.name

    def upload(self, item, text, meta):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.upload_to_cloud, f'{self.vault}/{item}/item', text)
            executor.submit(self.upload_to_cloud, f'{self.vault}/{item}/meta', json.dumps(meta))
    
    def delete_blob(self, blob):
        blob.delete()

    def delete(self):
        # Get all objects
        blobs = self.gcloud.list_blobs(prefix=self.vault)
        
        # Delete each object concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.delete_blob, blob): blob for blob in blobs}

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to delete blob: {e}")
        
    def get_vaults(self, vault: str = None):
        vault = self.vault if vault is None else vault
        blobs = self.gcloud.list_blobs(prefix=f'{vault}')
    
        directories = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) > 2 and not parts[1][0].isdigit():
                directories.add(parts[1])

        return list(directories)
