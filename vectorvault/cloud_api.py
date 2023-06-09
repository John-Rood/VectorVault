import requests

base_url = 'https://register.vectorvault.io'

def register(first_name: str, last_name: str, email: str, password: str) -> dict:
    '''
        Registers a new user:
        
        response = register('John', 'Smith', 'johnsmith@gmail.com', 'password')
        print(response)
    '''

    headers = {
        'Authorization': 'expected_authorization_code'  # replace with your actual authorization code
    }
    data = {
        'first': first_name,
        'last': last_name,
        'email': email,
        'password': password
    }
    response = requests.post(f'{base_url}/register', headers=headers, data=data)

    if response.status_code != 200:
        return {'error': response.text}

    return response.json()


def get_new_key(email, password):
    '''
        Enter your email and password to get a new api key:

        response = call_generate_new_key('user_id_here', 'password_here')
        print(response)
    '''

    headers = {
        'Authorization': 'expected_authorization_code'  # replace with your actual authorization code
    }
    data = {
        'email': email,
        'password': password,
    }
    # POST the data
    response = requests.post(f'{base_url}/generate_new_key', headers=headers, data=data)
    # Convert the response to JSON
    response_json = response.json()
    return response_json

def delete_key(email, api_key):
    '''
        Enter your email and the api key you wish to delete:

        response = call_delete_key('user_id_here', 'api_key_here')
        print(response)
    '''

    headers = {
        'Authorization': 'expected_authorization_code'  # replace with your actual authorization code
    }
    data = {
        'email': email,
        'api_key': api_key,
    }
    # POST the data
    response = requests.post(f'{base_url}/delete_key', headers=headers, data=data)
    # Convert the response to JSON
    response_json = response.json()
    return response_json
