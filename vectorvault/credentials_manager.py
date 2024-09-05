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

from google.auth.credentials import Credentials
import requests
import datetime

class CredentialsManager(Credentials):
    def __init__(self, user, api):
        self.user = user
        self.api = api
        self.token = None
        self.expiry = None
        self._universe_domain = "googleapis.com"
        self.refresh(requests.Request())

    def apply(self, headers, token=None):
        headers["Authorization"] = f"Bearer {self.token}"

    @property
    def valid(self):
        if self.expiry is None:
            return False
        return datetime.datetime.utcnow() < self.expiry

    def refresh(self, request):
        if not self.valid:
            data = {
                "user_id": self.user,
                "api_key": self.api,
            }
            response = requests.post('https://credentials.vectorvault.io/access', json=data)
            response.raise_for_status()  
            response_data = response.json()
            self.token = response_data['access_token']
            self.expiry = datetime.datetime.fromisoformat(response_data['expiry'])

    def before_request(self, request, method, url, headers):
        self.apply(headers)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
