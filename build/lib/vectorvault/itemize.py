import datetime 

def itemize(vault, x, meta=None, text=None, name=None):
    meta = {} if meta is None else meta

    if 'name' not in meta and name is None:
        name = f'{vault}-{x}'
    elif name is not None:
        name = __name__
    metaize(meta, name, x)
    return package(text, meta)
    
def metaize(meta, name, x):
    meta['name'] = name
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

def build_base_path(vault, x):
    return f'{vault}/{x}'

def append_path_suffix(base_path, is_item, is_meta):
    if is_item:
        suffix = 'item'
    elif is_meta:
        suffix = 'meta'
    else:
        suffix = ''
    return f'{base_path}/{suffix}'

def name(vault, x, item=False, meta=False):
    base_path = build_base_path(vault, x)
    final_path = append_path_suffix(base_path, item, meta)
    return final_path

def name_vecs(vault):
    return f'{vault}.ann'

def get_item(item):
    item_id = item["meta"]["item_id"]
    item_text = item["text"]
    item_meta = item["meta"]
    return item_text, item_id, item_meta

def build_return(results, item_data, meta):
    result = {
        "data": item_data,
        "metadata": meta
    }
    results.append(result)
