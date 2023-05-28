import requests

def register(first_name: str, last_name: str, email: str, password: str) -> dict:
    """
    Registers a new user.
    register('John', 'Smith', 'johnsmith@gmail.com', 'password')
    Args:
        first_name: First name of the user.
        last_name: Last name of the user.
        email: Email of the user.
        password: Password of the user.
    
    Returns:
        Response from the server.
    """
    url = 'https://vv-register-etrszydrna-uc.a.run.app/register'
    headers = {
        'Authorization': 'expected_authorization_code'  # replace with your actual authorization code
    }
    data = {
        'first': first_name,
        'last': last_name,
        'email': email,
        'password': password
    }
    response = requests.post(url, headers=headers, data=data)

    if response.status_code != 200:
        return {'error': response.text}

    return response.json()
