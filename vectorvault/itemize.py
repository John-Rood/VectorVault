import datetime 
from .vecreq import call_name_vecs, call_buildpath
from annoy import AnnoyIndex
import threading
from copy import deepcopy

def itemize(vault, x, meta=None, text=None, name=None):
    meta = deepcopy(meta) if meta else {}
    if 'name' not in meta and name is not None:
        meta['name'] = name
    elif name is None:
        meta['name'] = f'{vault}-{x}'
    metaize(meta, name, x)
    return package(text, meta)
    
def metaize(meta, name, x):
    meta['name'] = meta.get('name', name)
    meta['item_id'] = x  
    if 'created_at' not in meta:
        meta['created_at'] = datetime.datetime.utcnow().isoformat()
    if 'updated_at' not in meta:
        meta['updated_at'] = datetime.datetime.utcnow().isoformat()

def package(text, meta):
    item = {
        "text": text,
        "meta": meta
    }
    return item

def append_path_suffix(base_path, is_item, is_meta):
    if is_item:
        suffix = 'item'
    elif is_meta:
        suffix = 'meta'
    else:
        suffix = ''
    return f'{base_path}/{suffix}'

def cloud_name(v, x, user_id, api_key, item=False, meta=False):
    base_path = f'{v}/{x}'
    t = threading.Thread(target=call_buildpath, args=(v, x, user_id, api_key))
    t.start()
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

