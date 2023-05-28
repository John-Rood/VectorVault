import requests
import json

def call_name_vecs(vault, user_id, api_key, bytesize=None):
    url = f'{base1}/name_vecs'
    headers = {'Content-Type': 'application/json'}
    if bytesize:
        data = {
            "vault": vault,
            "user": user_id,
            "bytesize": bytesize,
            "api_key": api_key
            }
    else:
        data = {
            "vault": vault,
            "user": user_id,
            "api_key": api_key
            }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return json.loads(response.text)['result']

base1 = "https://get-vecs-etrszydrna-uc.a.run.app"
base2 = "https://get-loc-etrszydrna-uc.a.run.app"

def call_buildpath(v, x, user_id, api_key, bytesize=None):
    url = f'{base1}/buildpath'
    headers = {'Content-Type': 'application/json'}
    print(user_id)
    print(api_key)
    data = {
        "v": v,
        "x": x,
        "user": user_id,
        "api_key": api_key
    }
    if bytesize:
        data["bytesize"] = bytesize
    response = requests.post(url, headers=headers, json=data)

def call_proj():
    return 'vectorvault-361ab'

def call_vec(api_key):
    url = f'{base1}/85848'
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    return json.loads(response.text)['result']

def call_vecid(api_key):
    url = f'{base1}/94993838'
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    return json.loads(response.text)['result']

def call_get_vaults(user, api_key, vault=None):
    url = f"{base2}/vaults"
    params = {
        'user': user,
        'vault': vault,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()['vaults']

def call_get_total_vectors(user, vault, api_key):
    url = f"{base2}/total_vectors"
    params = {
        'user': user,
        'vault': vault,
        'api_key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()['total_vectors']

def call_items_by_vector(user, vault, vector, api_key, num_items=4):
    url = f"{base2}/items_by_vector"
    payload = {
        'user': user,
        'vault': vault,
        'vector': vector,
        'num_items': num_items,
        'api_key': api_key
    }
    response = requests.post(url, json=payload)
    return response.json()['results']

