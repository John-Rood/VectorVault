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

from .cloud_api import CloudAPI
import faiss
import numpy as np
from copy import deepcopy
import datetime
import time
import json


class FAISSIndex:
    """FAISS-based vector index with Annoy-compatible interface."""
    
    def __init__(self, dims, metric='angular'):
        self.dims = dims
        self.metric = metric
        # Use IndexIDMap to support non-sequential IDs
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dims))
        self._vectors = {}  # Store original (non-normalized) vectors by id
        self._built = False
    
    @property
    def ntotal(self):
        """Compatibility property for FAISS index ntotal."""
        return self.index.ntotal if self.index else 0
    
    def add_item(self, i, vector):
        """Add a vector at index i."""
        self._vectors[i] = np.array(vector, dtype=np.float32)
    
    def build(self, n_trees=10):
        """Build the index."""
        if not self._vectors:
            self._built = True
            return
        
        # Sort by ID to maintain order
        ids = sorted(self._vectors.keys())
        vectors_list = [self._vectors[i] for i in ids]
        
        vectors_np = np.array(vectors_list, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors_np)
        
        # Reset index and add all vectors
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dims))
        self.index.add_with_ids(vectors_np, ids_np)
        
        self._built = True
    
    def get_nns_by_vector(self, vector, n, include_distances=False, search_k=-1):
        """Find n nearest neighbors."""
        if self.index.ntotal == 0:
            return ([], []) if include_distances else []
        
        query = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(query)
        
        n = min(n, self.index.ntotal)
        distances, indices = self.index.search(query, n)
        
        # Filter out -1 indices (not found)
        result_indices = [int(idx) for idx in indices[0] if idx >= 0]
        
        if include_distances:
            # Convert similarity to angular distance: sqrt(2 * (1 - similarity))
            angular_distances = [np.sqrt(max(0, 2 * (1 - d))) for d in distances[0][:len(result_indices)]]
            return result_indices, angular_distances
        return result_indices
    
    def get_item_vector(self, i):
        """Get vector at index i."""
        if i in self._vectors:
            return self._vectors[i].tolist()
        return None
    
    def get_n_items(self):
        """Get number of items."""
        return len(self._vectors)
    
    def save(self, filename):
        """Save index and vectors to file."""
        filename = str(filename)
        # Save everything in one npz file (more portable than faiss.write_index)
        ids = list(self._vectors.keys())
        vectors = [self._vectors[i] for i in ids]
        # np.savez_compressed adds .npz extension if not present
        # To ensure consistent behavior, we explicitly add .npz for loading
        save_path = filename if filename.endswith('.npz') else filename + '.npz'
        np.savez_compressed(save_path,
                           dims=self.dims,
                           ids=np.array(ids, dtype=np.int64),
                           vectors=np.array(vectors, dtype=np.float32) if vectors else np.array([], dtype=np.float32).reshape(0, self.dims))
        # Copy to original filename if different (for backwards compatibility with .ann extension)
        if save_path != filename:
            import shutil
            shutil.copy(save_path, filename)
    
    def load(self, filename):
        """Load index from file."""
        filename = str(filename)
        # Try loading as npz first, then try with .npz extension added
        try:
            data = np.load(filename)
        except:
            data = np.load(filename + '.npz')
        
        self.dims = int(data['dims'])
        ids = data['ids']
        vectors = data['vectors']
        
        self._vectors = {}
        for i, vec in zip(ids, vectors):
            self._vectors[int(i)] = vec
        
        # Rebuild the FAISS index
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dims))
        if len(self._vectors) > 0:
            self.build(10)
        self._built = True

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
    base_path = CloudAPI._static_call_buildpath(user_id, api_key, v, x)
    final_path = append_path_suffix(base_path, item, meta)
    return final_path

def name_vecs(vault, user_id, api_key, byte=None):
    return CloudAPI._static_call_name_vecs(user_id, api_key, vault, byte)

def name_map(vault, user_id, api_key, byte=None):
    if user_id != user_id or api_key != api_key:
        raise('Check API key or user_id')
    extension = ''.join(map(chr, [46, 106, 115, 111, 110]))  
    vault_chars = [x for x in vault] 
    filename = ''.join(vault_chars) + extension  
    identity = lambda x: x  
    return identity(filename) 

def get_vectors(dims):
    """Return a FAISSIndex with Annoy-compatible interface."""
    return FAISSIndex(dims, 'angular')

def get_item(item):
    item_id = item["meta"]["item_id"]
    item_text = item["text"]
    item_meta = item["meta"]
    return item_text, item_id, item_meta

def build_return(item_data, meta, distance=None):
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
    return result

def get_time_statement(now, message_time):
    diff = now - message_time
    days, seconds = diff.days, diff.seconds
    human_readable_time = ""

    if days >= 365:
        years = days // 365
        human_readable_time = f"{years} {'year' if years == 1 else 'years'} ago: "
    elif days >= 30:
        months = days // 30
        human_readable_time = f"{months} {'month' if months == 1 else 'months'} ago: "
    elif days >= 1:
        human_readable_time = f"{days} {'day' if days == 1 else 'days'} ago: "
    elif seconds >= 3600:
        hours = seconds // 3600
        human_readable_time = f"{hours} {'hour' if hours == 1 else 'hours'} ago: "
    elif seconds >= 60:
        minutes = seconds // 60
        human_readable_time = f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago: "
    else:
        human_readable_time = "just now: "

    return human_readable_time

def load_json(json_object):
    """
    Loads a JSON object from either a string (potentially double-wrapped) or a direct object.
    Handles cases where the JSON might be escaped inside a string by attempting multiple parses.
    """
    def parse_json(string):
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided.")

    # If it's not a string, return as is
    if not isinstance(json_object, str):
        return json_object

    # Initial parse
    result = parse_json(json_object)

    # If the result is still a string (double-wrapped), parse again
    if isinstance(result, str):
        result = parse_json(result)

    return result