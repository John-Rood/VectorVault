from google.auth.credentials import Credentials
import requests
import datetime

class CustomCredentials(Credentials):
    def __init__(self, user, api):
        self.user = user
        self.api = api
        self.token = None
        self.expiry = None
        self.refresh(requests.Request())

    def apply(self, headers, token=None):
        headers["Authorization"] = f"Bearer {self.token}"

    @property
    def valid(self):
        if self.expiry is None:
            return False
        return datetime.datetime.now() < self.expiry

    def refresh(self, request):
        if not self.valid:
            data = {
                "user_id": self.user,
                "api_key": self.api,
            }
            response = requests.post('https://vv-creds-etrszydrna-uc.a.run.app/access', json=data)
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
