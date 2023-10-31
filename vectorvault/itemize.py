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
# from Vector Vault. See license for consent.import datetime 

from .vecreq import call_name_vecs, call_buildpath
from annoy import AnnoyIndex
import threading
from copy import deepcopy
import datetime
import time

def itemize(vault, x, meta=None, text=None, name=None):
    meta = deepcopy(meta) if meta else {}
    # If 'name' is not present in meta but name is provided
    if not meta.get('name') and name:
        meta['name'] = name
    # If 'name' is not present in meta and name is not provided 
    elif not meta.get('name') and not name:
        meta['name'] = f'{vault}-{x}'

    metaize(meta, name, x)
    return package(text, meta)  
    
def metaize(meta, name, x):
    meta['name'] = meta.get('name', name)
    meta['item_id'] = x  
    if 'created' not in meta:
        meta['created'] = datetime.datetime.utcnow().isoformat()
    if 'updated' not in meta:
        meta['updated'] = datetime.datetime.utcnow().isoformat()
    meta['time'] = time.time()

def package(text, meta):
    return {'text': text, 'meta': meta}


def append_path_suffix(base_path, is_item, is_meta):
    if is_item:
        suffix = 'item'
    elif is_meta:
        suffix = 'meta'
    else:
        suffix = ''
    return f'{base_path}/{suffix}'

def cloud_name(v, x, user_id, api_key, item=False, meta=False):
    base_path = call_buildpath(v, x, user_id, api_key)
    final_path = append_path_suffix(base_path, item, meta)
    return final_path

def name_vecs(vault, user_id, api_key, byte=None):
    return call_name_vecs(vault, user_id, api_key, byte)

def get_vectors(dims):
    return AnnoyIndex(dims, 'angular')

def get_item(item):
    item_id = item["meta"]["item_id"]
    item_text = item["text"]
    item_meta = item["meta"]
    return item_text, item_id, item_meta

def build_return(results, item_data, meta, distance=None):
    if not distance:
        result = {
            "data": item_data,
            "metadata": meta
        }
    else:
        result = {
            "data": item_data,
            "metadata": meta,
            "distance": distance
        }
    results.append(result)
