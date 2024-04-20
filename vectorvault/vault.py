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

import numpy as np
import tempfile
import math
import os
import time
import uuid
import re
import json
import traceback
import random
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread as T
from .cloudmanager import CloudManager
from .ai import AI, openai
from .itemize import itemize, name_vecs, get_item, get_vectors, build_return, cloud_name, name_map, get_time_statement
from .vecreq import call_get_similar, call_cloud_save
from .tools_gpt import ToolsGPT


class Vault:
    def __init__(self, user: str = None, api_key: str = None, openai_key: str = None, vault: str = None, embeddings_model: str = None, verbose: bool = False, conversation_user_id: str = None):
        ''' 
        ### Create a vector database instance like this:
        ```
        vault = Vault(user='YOUR_EMAIL',
              api_key='VECTOR_VAULT_API_KEY',
              openai_key='OPENAI_API_KEY',
              vault='VAULT_NAME',
              verbose=True)
        ```

        ### Add data to the vector database aka "the Vault":
        ```
        vault.add('some text')
        vault.get_vectors()
        vault.save()
        ```

        ### Basic ChatGPT response:
        `basic_chatgpt_answer = vault.get_chat('some question')`

        ### Advanced RAG response: (Retrieval Augemented Generation) -> (uses 'get_context=True' to reference the Vault - [which will internally pulls 4 vector similar search results from the database to be used as context before responding])
        `rag_answer = vault.get_chat('some question', get_context=True)`

        ### Change the model to GPT4 with the `model` param in get_chat:
        `gpt4_rag_answer = vault.get_chat('some question', get_context=True, model='gpt-4')`

        ### Change the personality of your bot with the personality_message:
        ```
        vault = Vault(user='YOUR_EMAIL',
              api_key='VECTOR_VAULT_API_KEY',
              openai_key='OPENAI_API_KEY',
              vault='VAULT_NAME',
              personality_message='Answer like you are Snoop Dogg',
              verbose=False)
        ```
        '''
        self.user = user.lower()
        self.vault = vault.strip() if vault else 'home'
        self.api = api_key
        self.verbose = verbose
        self.embeddings_model = embeddings_model if embeddings_model else 'text-embedding-3-small'
        self.dims = 1536 if embeddings_model != 'text-embedding-3-large' else 3072
        self.cloud_manager = CloudManager(self.user, self.api, self.vault)
        if self.verbose:
            print(f'Connected vault: {self.vault}')

        if openai_key:
            self.openai_key = openai_key
            openai.api_key = self.openai_key
        self.vectors = get_vectors(self.dims)
        self.x = 0
        self.x_checked = False
        self.x_loaded_checked = False
        self.vecs_loaded = False
        self.map = {}
        self.items = []
        self.last_time = None
        self.saved_already = False
        self.ai_loaded = False
        self.tools = ToolsGPT(verbose=verbose)
        self.rate_limiter = RateLimiter(max_attempts=30)
        self.cuid = conversation_user_id

    def get_vaults(self, vault: str = None):
        '''
            Returns a list of vaults within the current vault directory 
        '''
        vault = self.vault if vault is None else vault
        if self.cloud_manager is None:
           time.sleep(.3)
        return self.cloud_manager.list_vaults(vault)


    def get_total_items(self, vault: str = None):
        '''
            Returns the total number of vectored items in the Vault
        '''
        if not vault:
            self.load_mapping()
            return len(self.map)
        else:
            try:
                temp_file_path = self.cloud_manager.download_to_temp_file(name_map(vault, self.user, self.api))
                with open(temp_file_path, 'r') as json_file:
                    _map = json.load(json_file)
                os.remove(temp_file_path)
                return len(_map)
            except: # it doesn't exist
                return 0


    def get_tokens(self, text: str):
        '''
            Returns the number of tokens for any given text
        '''
        self.load_ai() if not self.ai_loaded else None
        return self.ai.get_tokens(text)
    

    def get_distance(self, id1: int, id2: int):
        '''
            Returns the distance between two vectors - item ids are needed to compare
        '''
        self.check_index_loaded()
        return self.vectors.get_distance(id1, id2)
    

    def get_item_vector(self, item_id: int):
        '''
            Returns the vector from an item id
        '''
        self.check_index_loaded()
        return self.vectors.get_item_vector(item_id)


    def load_ai(self):
        self.ai_loaded = True
        self.ai = AI(verbose=self.verbose)
        self.ai.context_prompt = self.fetch_custom_prompt()
        self.ai.prompt = self.fetch_custom_prompt(context=False)
        self.ai.personality_message = self.fetch_personality_message()


    def save_personality_message(self, text: str):
        '''
            Saves personality_message to the vault and use it by default from now on
        '''
        self.cloud_manager.upload_personality_message(text)

        print(f"Personality message saved") if self.verbose else 0
            

    def fetch_personality_message(self):
        '''
            Retrieves personality_message from the vault if it is there or else use the defualt
        '''
        try:
            personality_message = self.cloud_manager.download_text_from_cloud(f'{self.vault}/personality_message')
        except:
            if self.ai_loaded:
                personality_message = self.ai.personality_message
            else: # only when called externally in some situations
                self.load_ai()
                personality_message = self.ai.personality_message
                
        return personality_message
    

    def save_custom_prompt(self, text: str, context=True):
        '''
            Saves custom_prompt to the vault and use it by default from now on
            Param: "context" True = context prompt ; False = main prompt
        '''
        self.cloud_manager.upload_custom_prompt(text) if context else self.cloud_manager.upload_no_context_prompt(text)

        print(f"Custom prompt saved") if self.verbose else 0
            

    def fetch_custom_prompt(self, context=True):
        '''
            Retrieves custom_prompt from the vault if there or eles use defualt - (used for get_context = True responses)
            context == False will return custom prompt for not context situations
        '''
        try:
            prompt = self.cloud_manager.download_text_from_cloud(f'{self.vault}/prompt') if context else self.cloud_manager.download_text_from_cloud(f'{self.vault}/no_context_prompt')
        except:
            if self.ai_loaded:
                prompt = self.ai.main_prompt_with_context if context else self.ai.prompt
            else: # only when called externally in some situations
                self.load_ai()
                prompt = self.ai.main_prompt_with_context if context else self.ai.prompt
            
        return prompt


    def save(self, trees: int = 10, vault = None):
        '''
            Saves all the data added locally to the Cloud. All Vault references are Cloud references.
            To add data to your Vault and access it later, you must first call add(), then get_vectors(), and finally save().
        '''
        vault = vault if vault else self.vault
        if self.saved_already:
            self.clear_cache()
            print("The last save was aborted before the build process finished. Clearing cache to start again...")
        self.saved_already = True # Make sure the if the save process is interrupted, data will not get corrupted
        start_time = time.time()
        self.vectors.build(trees)

        total_saved_items = 0

        with ThreadPoolExecutor() as executor:
            for item in self.items:
                item_text, item_id, item_meta = get_item(item)
                executor.submit(self.cloud_manager.upload, self.map.get(str(item_id)), item_text, item_meta, vault)
                total_saved_items += 1

        self.upload_vectors(vault)
        print(f"upload time --- {(time.time() - start_time)} seconds --- {total_saved_items} items saved") if self.verbose else 0
            

    def clear_cache(self):
        '''
            Clears the cache for all the loaded items 
        '''
        self.reload_vectors()
        self.vecs_loaded = True
        self.saved_already = False


    def delete(self):
        '''
            Deletes the entire Vault and all contents
        '''
        print('Deleting started. Note: this can take a while for large datasets') if self.verbose else 0
            
        # Clear the local vector data
        self.vectors = get_vectors(self.dims)
        self.items.clear()
        self.cloud_manager.delete()
        self.x = 0
        print('Vault deleted')
    

    def remap(self, item_id):
        for i in range(item_id, len(self.map) - 1):
            self.map[str(i)] = self.map[str(i + 1)]
        self.map.popitem()


    def update_vault_data(self):
        nary = []
        try:
            vault_data = self.cloud_manager.get_mapping()
            all_vaults = self.get_vaults('')
            _data = [vault for vault in vault_data if vault['vault'] in all_vaults]
            for i in all_vaults:
                total_items = self.get_total_items(i)
                time.sleep(.01)
                item = self.get_items([ total_items - 1 ], i)[0]['metadata']
                try: 
                    time_for_last_item = item['updated']
                except:
                    time_for_last_item = item['updated_at']

                vault_dict = { 'vault': f'{i}', 'total_items': total_items, 'last_update': time_for_last_item }

                for original_data in _data:  
                    if original_data['vault'] == f'{i}':
                        if 'last_use' in original_data:
                            vault_dict['last_use'] = original_data['last_use']
                        if 'total_use' in original_data:
                            vault_dict['total_use'] = original_data['total_use']

                nary.append(vault_dict)

        except Exception as e:
            print(f"An error occurred: {e}")
            nary.append({ 'vault': 'demo', 'total_items': 0, 'last_update': time.time(), 'last_use': time.time(), 'total_use': 0 })

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            nary_temp_file_path = temp_file.name
            json.dump(nary, temp_file, indent=2)

        self.cloud_manager.upload_temp_file(nary_temp_file_path, f'{self.cloud_manager.username}.json')


    def delete_items(self, item_ids: List[int], trees: int = 10) -> None:
        '''
            Deletes one or more items from item_id(s) passed in.
            item_ids is an int or a list of integers
        '''
        def rebuild_vectors(item_id):
            '''Deletes target vector and rebuilds the database'''
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)
            plus_one = 0

            def download_and_upload(i, plus_one):
                if plus_one == 1:
                    metadata = json.loads(self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.old_map[str(i)], self.user, self.api, meta=True)))
                    metadata['item_id'] = i - plus_one
                    self.cloud_manager.upload_to_cloud(cloud_name(self.vault, self.old_map[str(i - plus_one)], self.user, self.api, meta=True), json.dumps(metadata))

            with ThreadPoolExecutor() as executor:
                for i in range(num_existing_items):
                    if i == item_id: # the item being deleted 
                        plus_one += 1 
                    else: 
                        vector = self.vectors.get_item_vector(i - plus_one) 
                        new_index.add_item(i - plus_one, vector)
                        executor.submit(download_and_upload, i, plus_one)

            self.vectors = new_index
            self.vectors.build(trees)
            self.upload_vectors()

        item_ids = [item_ids] if isinstance(item_ids, int) else item_ids
        for item_id in item_ids:
            self.load_vectors()
            self.cloud_manager.delete_item(self.map[str(item_id)])
            self.old_map = self.map
            self.remap(item_id)
            rebuild_vectors(item_id)

        print(f'Item {item_id} deleted') if self.verbose else 0


    def edit_item(self, item_id: int, new_text: str, trees: int = 10) -> None:
        '''
            Edits any item. Enter the new text and new vectors will automatically be created.
            New data and vectors will be uploaded and the old data will be deleted
        '''
        def edit_vector(item_id, new_vector):
            '''Replaces old vector with new content vector'''
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)

            for i in range(num_existing_items):
                if i == item_id:
                    new_index.add_item(i, new_vector)
                else: 
                    vector = self.vectors.get_item_vector(i) 
                    new_index.add_item(i, vector)

            self.vectors = new_index
            self.vectors.build(trees)
            self.upload_vectors()

        self.load_vectors()
        self.cloud_manager.upload_to_cloud(cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, item=True), new_text)
        edit_vector(item_id, self.process_batch([new_text], never_stop=False, loop_timeout=180)[0])

        print(f'Item {item_id} edited') if self.verbose else 0


    def edit_item_meta(self, item_id: int, metadata) -> None:
        '''
            Edit and save any item's metadata
        '''
        self.cloud_manager.upload_to_cloud(cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, meta=True), json.dumps(metadata))
        print(f'Item {item_id} metadata saved') if self.verbose else 0


    def check_index_loaded(self):
        if not self.x_loaded_checked:
            start_time = time.time()
            if self.cloud_manager.vault_exists(name_vecs(self.vault, self.user, self.api)):
                if not self.vecs_loaded:
                    self.load_vectors()

            self.x_loaded_checked = True
            print("initialize index --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0

    def check_index(self):
        if not self.x_checked:
            start_time = time.time()
            if self.cloud_manager.vault_exists(name_vecs(self.vault, self.user, self.api)):
                if self.vecs_loaded:
                    self.reload_vectors()

            self.x_checked = True
            print("initialize index --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
                

    def load_mapping(self, vault = None):
        '''Internal function only'''
        vault = vault if vault else self.vault 
        try: # try to get the map
            temp_file_path = self.cloud_manager.download_to_temp_file(name_map(vault, self.user, self.api))
            with open(temp_file_path, 'r') as json_file:
                self.map = json.load(json_file)
            os.remove(temp_file_path)
        except: # it doesn't exist
            if self.cloud_manager.vault_exists(name_vecs(vault, self.user, self.api)): # but if the vault does
                self.map = {str(i): str(i) for i in range(self.vectors.get_n_items())}

    def add_to_map(self):
        self.map[str(self.x)] = str(uuid.uuid4())
        self.x +=1
    

    def load_vectors(self, vault = None):
        start_time = time.time()
        vault = vault if vault else self.vault 
        t = T(target=self.load_mapping())
        t.start()
        temp_file_path = self.cloud_manager.download_to_temp_file(name_vecs(vault, self.user, self.api))
        self.vectors.load(temp_file_path)
        os.remove(temp_file_path)
        t.join()
        self.vecs_loaded = True
        self.x_checked = False
        print("get load vectors --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0


    def reload_vectors(self):
        num_existing_items = self.vectors.get_n_items()
        new_index = get_vectors(self.dims)
        count = -1
        for i in range(num_existing_items):
            count += 1
            vector = self.vectors.get_item_vector(i)
            new_index.add_item(i, vector)
        self.x = count + 1
        self.vectors = new_index
        self.vecs_loaded = False


    def make_3d_map(self, highlight_id: int = None, return_html: bool = False):
        from kneed import KneeLocator
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        from sklearn.manifold import TSNE
        import plotly.graph_objs as go

        def choose_eps(vectors_3d, k=4):
            # Use NearestNeighbors to find the k-nearest distances for each point
            neigh = NearestNeighbors(n_neighbors=k)
            neigh.fit(vectors_3d)
            distances, indices = neigh.kneighbors(vectors_3d)

            # Sort the distances
            sorted_distances = np.sort(distances[:, k-1], axis=0)

            # Find the knee point for the k-distance graph
            kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, S=1.0, curve='convex', direction='increasing')
            eps = sorted_distances[kneedle.knee]

            return eps
        
        self.load_vectors()
        # Retrieve all vectors from the index
        vectors = [self.vectors.get_item_vector(i) for i in range(self.vectors.get_n_items())]
        # Convert list of vectors to a NumPy array
        vectors_array = np.array(vectors)
        # Use t-SNE to project the vectors into 3D space
        tsne = TSNE(n_components=3, random_state=0)
        vectors_3d = tsne.fit_transform(vectors_array)
        eps_val = choose_eps(vectors_3d)
        print(eps_val)
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps_val, min_samples=10)  # These parameters may need tuning
        cluster_labels = dbscan.fit_predict(vectors_3d)

        # Generate hover text for each point
        hover_texts = [
            'ID: {} Cluster: {}'.format(i, 'Outlier' if lbl == -1 else lbl)
            for i, lbl in enumerate(cluster_labels)
            ]

        # Trace for all points with cluster coloring
        trace_all = go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=cluster_labels,  # Color by cluster label
                colorscale='Viridis',  # Color scale for clusters
                line=dict(width=0)  # No border around markers
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Clustered points'
        )

        data = [trace_all]

        # For the highlighted point, if it exists
        if highlight_id is not None:
            trace_highlight = go.Scatter3d(
                x=[vectors_3d[highlight_id, 0]],
                y=[vectors_3d[highlight_id, 1]],
                z=[vectors_3d[highlight_id, 2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',  # Highlighted point is red
                ),
                text=['Highlighted ID: {}'.format(highlight_id)],
                hoverinfo='text',
                name='Highlighted point'
            )
            data.append(trace_highlight)

        # Layout for the 3D plot
        layout = go.Layout(
            title='3D Visualization with DBSCAN Clustering',
            scene=dict(
                xaxis=dict(title='Dimension 1'),
                yaxis=dict(title='Dimension 2'),
                zaxis=dict(title='Dimension 3'),
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)
        if return_html:
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        else:
            fig.show()


    def upload_vectors(self, vault = None):
        vault = vault if vault else self.vault
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            vector_temp_file_path = temp_file.name
            self.vectors.save(vector_temp_file_path)
            byte = os.path.getsize(vector_temp_file_path)
            self.cloud_manager.upload_temp_file(vector_temp_file_path, name_vecs(vault, self.user, self.api, byte))
        
        map_temp_file_path = None
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            map_temp_file_path = temp_file.name
            json.dump(self.map, temp_file, indent=2)
        
        self.cloud_manager.upload_temp_file(map_temp_file_path, name_map(vault, self.user, self.api, byte))
        if os.path.exists(map_temp_file_path):
            os.remove(map_temp_file_path)
            
        self.items.clear()
        self.vectors = get_vectors(self.dims)
        self.x_checked = False
        self.vecs_loaded = False
        self.saved_already = False


    def split_text(self, text: str, min_threshold: int = 1000, max_threshold: int = 16000):
        '''
        Internal function
        Splits the given text into chunks of sentences such that each chunk's length 
        is at least min_threshold characters but does not exceed max_threshold characters.
        Sentences are not broken mid-way.
        '''
        segments = []
        sentence_spans = list(re.finditer(r"(?<=[.!?])\s+", text))
        
        current_segment = []
        current_length = 0
        sentence_start = 0

        for sentence_span in sentence_spans:
            sentence = text[sentence_start:sentence_span.end()]

            if current_length + len(sentence) > max_threshold:
                if current_segment:
                    segments.append(" ".join(current_segment))
                current_segment = [sentence]
                current_length = len(sentence)
            else:
                current_segment.append(sentence)
                current_length += len(sentence)

            if current_length >= min_threshold:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_length = 0

            sentence_start = sentence_span.end()

        # Add the remaining sentences or partial sentences to the last segment
        last_sentence = text[sentence_start:]
        if last_sentence:
            current_segment.append(last_sentence)
        
        # Ensure that even the last segment is long enough if there are prior segments
        if current_segment and (current_length >= min_threshold or not segments):
            segments.append(" ".join(current_segment))

        print(f'split_text chunks: {len(segments)}') if self.verbose else 0
        
        return segments
    

    def get_items(self, ids: List[int] = [], vault: str = None) -> list:
        '''
            Get one or more items from the database. 
            Input the item id(s) in a list. -> Returns the items 

            - Example Single Item Usage:
                item = vault.get_items([132])

            - Example Multi-Item Usage:
                items = vault.get_items([132, 128, 393, 74, 644, 71])

            Sample return when called:
            `[
                {
                'data': 'The Project Gutenberg eBook of The Prince...',
                'metadata': 
                    {
                        'name': 'vaultname-0',
                        'item_id': 0,
                        'created': '2023-12-28T19:55:26.406048',
                        'updated': '2023-12-28T19:55:26.406053',
                        'time': 1703793326.4060538
                    }
                }
            ]`
        '''
        results = [None] * len(ids)  # Pre-fill the results list with placeholders
        start_time = time.time()
        
        def fetch_item(index, i, _map):
            # Function to fetch a single item, to be run in parallel
            item_data = self.cloud_manager.download_text_from_cloud(cloud_name(vault, _map[str(i)], self.user, self.api, item=True))
            meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(vault, _map[str(i)], self.user, self.api, meta=True))
            meta = json.loads(meta_data) 
            return index, build_return(item_data, meta)

        if vault: # if custom vault, get custom mapping
            temp_file_path = self.cloud_manager.download_to_temp_file(name_map(vault, self.user, self.api))
            with open(temp_file_path, 'r') as json_file:
                _map = json.load(json_file)
            os.remove(temp_file_path)
        else:
            self.load_mapping()
            vault = self.vault
            _map = self.map

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_item, i, id, _map) for i, id in enumerate(ids)]
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result  # Insert each result at its corresponding index
            
        print(f"Retrieved {len(ids)} items --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
        
        return results


    
    def get_items_by_vector(self, vector: list, n: int = 4, include_distances: bool = False):
        '''
            Internal function that returns vector similar items. Requires input vector, returns similar items
        '''
        try:
            self.load_vectors()
            start_time = time.time()
            if not include_distances:
                vecs = self.vectors.get_nns_by_vector(vector, n)
                results = [None] * len(vecs)  # Pre-fill the results list with placeholders

                def fetch_item(index, vec):
                    # Function to fetch a single item, to be run in parallel
                    item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, item=True))
                    meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, meta=True))
                    meta = json.loads(meta_data)
                    return index, build_return(item_data, meta)
                
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(fetch_item, i, vec) for i, vec in enumerate(vecs)]
                    for future in as_completed(futures):
                        index, result = future.result()
                        results[index] = result  # Insert each result at its corresponding index

                if self.verbose:
                    print(f"get {n} items back --- %s seconds ---" % (time.time() - start_time))
                return results
            else:
                vecs, distances = self.vectors.get_nns_by_vector(vector, n, include_distances=include_distances)
                results = [None] * len(vecs)  # Pre-fill the results list with placeholders

                def fetch_item(index, vec, distance):
                    # Function to fetch a single item, to be run in parallel
                    item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, item=True))
                    meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(vec)], self.user, self.api, meta=True))
                    meta = json.loads(meta_data)
                    return index, build_return(item_data, meta, distance)

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(fetch_item, i, vec, distances[i]) for i, vec in enumerate(vecs)]
                    for future in as_completed(futures):
                        index, result = future.result()
                        results[index] = result  # Insert each result at its corresponding index

                if self.verbose:
                    print(f"get {n} items back --- %s seconds ---" % (time.time() - start_time))
                return results
        except:
            return [{'data': 'No data has been added', 'metadata': {'no meta': 'No metadata has been added'}}]


    def get_similar_local(self, text: str, n: int = 4, include_distances: bool = False):
        '''
            Returns similar items from the Vault as the one you entered, but locally
            (saves a few milliseconds and is sometimes used on production builds)
        '''
        # self.cloud_manager.update() if not self.cuid else None
        vector = self.process_batch([text], never_stop=False, loop_timeout=180)[0]
        return self.get_items_by_vector(vector, n, include_distances = include_distances)
    

    def get_similar(self, text: str, n: int = 4, include_distances: bool = False, vault = None):
        '''
            Returns similar items from the Vault as the text you enter.

            Param `include_distances = True` adds the "distance" field to the return.
            The distance can be useful for assessing similarity differences in the items returned. 
            Each item has its' own distance number, and this changes the structure of the output.
        '''
        return call_get_similar(self.user, vault if vault else self.vault, self.api, self.openai_key, text, n, include_distances=include_distances, verbose=self.verbose, embeddings=self.embeddings_model)


    def add_item(self, text: str, meta: dict = None, name: str = None):
        """
            If your text length is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        self.check_index()
        new_item = itemize(self.vault, self.x, meta, text, name)
        self.items.append(new_item)
        self.add_to_map()

    def add(self, text: str, meta: dict = None, name: str = None, split: bool = False, split_size: int = 1000, max_threshold: int = 16000):
        """
            If your text length is greater than 4000 tokens, Vault.split_text(your_text)  
            will automatically be added
        """
        self.check_index()

        if len(text) > 15000 or split:
            if self.verbose:
                print('Using the built-in "split_text()" function to get a list of texts') 
            texts = self.split_text(text, min_threshold=split_size, max_threshold=max_threshold) # returns list of text segments
        else:
            texts = [text]
        for text in texts:
            self.add_item(text, meta, name)

    def add_n_save(self, text: str, meta: dict = None, name: str = None, split: bool = False, split_size: int = 1000, max_threshold: int = 16000):
        """
            Adds data, gets vectors, then saves the data to the cloud in one call
            If your text length is greater than 4000 tokens, your text will automatically be split by
            Vault.split_text(your_text).
        """
        self.add(text=text, meta=meta, name=name, split=split, split_size=split_size, max_threshold=max_threshold)
        self.get_vectors()
        self.save()
        T(target=self.clear_cache).start()


    def add_item_with_vector(self, text: str, vector: list, meta: dict = None, name: str = None):
        """
            If your text length is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        self.check_index()
        start_time = time.time()

        if self.ai.get_tokens(text) > 4000:
            raise 'Text length too long. Use the "split_text() function to get a list of text segments'

        # Add vector to vectorspace
        self.vectors.add_item(self.x, vector)
        self.items.append(itemize(self.vault, self.x, meta, text, name))
        self.add_to_map()

        print("add item time --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
            

    def process_batch(self, batch_text_chunks, never_stop, loop_timeout):
        '''
            Internal function
        '''
        loop_start_time = time.time()
        exceptions = 0
        while True:
            try:
                res = openai.embeddings.create(input=batch_text_chunks, model=self.embeddings_model)
                break
            except Exception as e:
                last_exception_time = time.time()
                exceptions = 1 if time.time() - last_exception_time > 180 else + 1
                print(f"API Error: {e}. Sleeping {(exceptions * 5)} seconds")
                time.sleep((exceptions * 5))

                if not never_stop or (time.time() - loop_start_time) > loop_timeout:
                    try:
                        res = openai.embeddings.create(input=batch_text_chunks, model=self.embeddings_model)
                        break
                    except Exception as e:
                        if exceptions >= 5:
                            print(f"API has failed for too long. Exiting loop with error: {e}.")
                            break
                        raise TimeoutError("Loop timed out")
        return [record.embedding for record in res.data]
        

    def get_vectors(self, batch_size: int = 32, never_stop: bool = False, loop_timeout: int = 777):
        '''
            Takes text data added to the vault, and gets vectors for them
        '''
        self.check_index()
        start_time = time.time()

        # If last_time isn't set, assume it's a very old time (e.g., 10 minutes ago)
        if not self.last_time:
            self.last_time = start_time - 600

        texts = [item['text'] for item in self.items]
        num_batches = int(np.ceil(len(texts) / batch_size))
        batches_text_chunks = [
            texts[i * batch_size:min((i + 1) * batch_size, len(texts))]
            for i in range(num_batches)
        ]

        batch_embeddings_list = [self.process_batch(batch_text_chunk, never_stop=never_stop, loop_timeout=loop_timeout) for batch_text_chunk in batches_text_chunks]

        current_item_index = 0
        for batch_embeddings in batch_embeddings_list:
            for embedding in batch_embeddings:
                item_index = self.items[current_item_index]["meta"]["item_id"]
                self.vectors.add_item(item_index, embedding)
                current_item_index += 1

        self.last_time = time.time()
        print("get vectors time --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
            

    def get_chat(self, text: str = None, history: str = '', summary: bool = False, get_context: bool = False, 
                 n_context: int = 4, return_context: bool = False, history_search: bool = False, smart_history_search: bool = False, 
                 model: str = 'gpt-3.5-turbo', include_context_meta: bool = False, custom_prompt: bool = False, 
                 local: bool =False, temperature: int = 0, timeout: int = 300):
        '''
            Chat get response from OpenAI's ChatGPT. 
            Models: ChatGPT = "gpt-3.5-turbo" • GPT4 = "gpt-4" 
            Large Context Models: ChatGPT 16k = "gpt-3.5-turbo-16k" • GPT4 32k = "gpt-4-32k"
            Best Versions: "gpt-3.5-turbo-0301" is March 2023 ChatGPT (best version) - "gpt-4-0314" (best version)

            include_context_meta (bool): If True, will also include context metadata, rather than just the context text string

            Rate limiting, auto retries, and chat histroy slicing built-in so you can chat with ease. 
            Enter your text, add optional chat history, and optionally choose a summary response (default: summmary = False)

            - Example Signle Usage: 
            `response = vault.get_chat(text)`

            - Example Chat: 
            `response = vault.get_chat(text, chat_history)`
            
            - Example Summary: 
            `summary = vault.get_chat(text, summary=True)`

            - Example Context-Based Response:
            `response = vault.get_chat(text, get_context = True)`

            - Example Context-Based Response w/ Chat History:
            `response = vault.get_chat(text, chat_history, get_context = True)`

            - Example Context-Response with Context Samples Returned:
            `vault_response = vault.get_chat(text, get_context = True, return_context = True)`
            
            Response is a string, unless return_context == True, then response will be a dictionary 

            - Example to print dictionary results:
            # print response:
            `print(vault_response['response'])` 

            # print context:
            for item in answer['context']:
                print("\n\n", f"item {item['metadata']['item_id']}")
                print(item['data'])

            smart_history_search is False by default skip adding the history of the conversation to the text input for similarity search (useful if history contains subject infomation useful for answering the new text input and the text input doesn't contain that info)
            
            - Example Custom Prompt:
            `response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

            `custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
            `llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. 

            *Example WIHTOUT Vault Context:*

            ```python
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
            ```

            Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
            `llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

            *Example WITH Vault Context:*  
            ```python
            custom_prompt = """
                Use the following Context to answer the Question at the end. 
                Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

                Chat History (if any): {history}

                Additional Context: {context}

                Question: {question}

                (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
                Answer:
            """ 
            response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
            ```

        '''
        self.load_ai() if not self.ai_loaded else None
        model = model.lower()
        start_time = time.time()
    
        history = self.get_conversation_history(self.cuid, text) if self.cuid else history
        
        if text: 
            inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = [0]

        response = ''
        for segment in inputs:
            attempts = 0
            while True:
                try:
                    # Make API call
                    if summary and not get_context:
                        token_limit = self.ai.model_token_limits.get(model, 4000)
                        total_tokens = self.ai.get_tokens(segment)
                        response = ""

                        for i in range(math.ceil(total_tokens / token_limit)):
                            # Calculate start and end indices for the current chunk
                            start_index = i * token_limit
                            end_index = min((i + 1) * token_limit, total_tokens)
                            # Extract the current chunk from the segment
                            current_chunk = segment[start_index:end_index]
                            # Process the current chunk and concatenate the response
                            response += ' ' + self.ai.summarize(current_chunk, model=model, custom_prompt=custom_prompt, temperature=temperature)
                            self.rate_limiter.on_success()

                    elif text and get_context and not summary:
                        if smart_history_search:
                            custom_entry = f"Using the current message, with the message history, what subject is the user is focused on. \nCurrent message: {text}. \n\nPrevious messages: {history}."
                            search_input = self.ai.llm(custom_prompt=custom_entry, model=model, temperature=temperature, timeout=timeout)
                        else:
                            search_input = segment + history if history_search else segment
                            
                        if include_context_meta:
                            context = self.get_similar(search_input, n=n_context) if not local else self.get_similar_local(search_input, n=n_context)
                            input_ = str(context) # all of the metadata included
                        else:
                            context = self.get_similar(search_input, n=n_context) if not local else self.get_similar_local(search_input, n=n_context)
                            input_ = ''
                            for text in context:
                                input_ += text['data'] # just the text data, no metadata
                        response = self.ai.llm_w_context(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                    else: # Custom prompt only
                        if inputs[0] == 0:
                            response = self.ai.llm(model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                        else:
                            response = self.ai.llm(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                            
                    # If the call is successful, reset the backoff
                    self.rate_limiter.on_success()
                    break
                except Exception as e:
                    # If the call fails, apply the backoff
                    attempts += 1
                    print(traceback.format_exc())
                    print(f"API Error: {e}. Backing off for {self.rate_limiter.current_delay} seconds.")
                    self.rate_limiter.on_failure()
                    if attempts >= self.rate_limiter.max_attempts:
                        print(f"API Failed too many times, exiting loop: {e}.")
                        break

        self.update_conversation_history(self.cuid, f'User: {text} \n\nAI: {response}') if self.cuid else None
        print("get chat time --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
            
        return {'response': response, 'context': context} if return_context else response
        

    def get_chat_stream(self, text: str = None, history: str = '', summary: bool = False, get_context: bool = False,
                        n_context: int = 4, return_context: bool = False, history_search: bool = False, smart_history_search: bool = False, 
                        model: str ='gpt-3.5-turbo', include_context_meta: bool = False, metatag: bool = False,
                        metatag_prefixes: bool = False, metatag_suffixes: bool = False, custom_prompt: bool = False, 
                        local: bool = False, temperature: int = 0, timeout: int = 300):
        '''
            Always use this get_chat_stream() wrapped by either print_stream(), or cloud_stream().
            cloud_stream() is for cloud functions, like a flask app serving a front end elsewhere.
            print_stream() is for local console printing

            - Example Signle Usage: 
            `response = vault.print_stream(vault.get_chat_stream(text))`

            - Example Chat: 
            `response = vault.print_stream(vault.get_chat_stream(text, chat_history))`
            
            - Example Summary: 
            `summary = vault.print_stream(vault.get_chat_stream(text, summary=True))`

            - Example Context-Based Response:
            `response = vault.print_stream(vault.get_chat_stream(text, get_context = True))`

            - Example Context-Based Response w/ Chat History:
            `response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True))`

            - Example Context-Response with Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True))`

            - Example Context-Response with SPECIFIC META TAGS for Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author']))`
            
            - Example Context-Response with SPECIFIC META TAGS for Context Samples Returned & Specific Meta Prefixes and Suffixes:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author'], metatag_prefixes=['\n\n Title: ', '\nAuthor: '], metatag_suffixes=['', '\n']))`
            

            - Example Custom Prompt:
            `response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

            `custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
            `llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. 

            *Example WIHTOUT Vault Context:*

            ```python
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
            ```

            Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
            `llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

            *Example WITH Vault Context:*  
            ```python
            custom_prompt = """
                Use the following Context to answer the Question at the end. 
                Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

                Chat History (if any): {history}

                Additional Context: {context}

                Question: {question}

                (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
                Answer:
            """ 
            response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
            ```
        '''
        self.load_ai() if not self.ai_loaded else None
        model = model.lower()
        start_time = time.time()
    
        history = self.get_conversation_history(self.cuid, text) if self.cuid else history

        if text:
            inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = []
                
        counter = 0
        for segment in inputs:
            start_time = time.time()
            exceptions = 0
            full_response = ''
            while True:
                try:
                    if summary and not get_context:
                        try:
                            token_limit = self.ai.model_token_limits.get(model, 4000)
                            total_tokens = self.ai.get_tokens(segment)

                            for i in range(math.ceil(total_tokens / token_limit)):
                                # Calculate start and end indices for the current chunk
                                start_index = i * token_limit
                                end_index = min((i + 1) * token_limit, total_tokens)
                                # Extract the current chunk from the segment
                                current_chunk = segment[start_index:end_index]
                                # Process the current chunk and concatenate the response
                                for word in self.ai.summarize_stream(current_chunk, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                    full_response += word
                                    yield word
                                yield ' '

                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e
                        if counter == len(inputs):
                            self.update_conversation_history(self.cuid, f'User: {text} \n\nAI: {full_response}') if self.cuid else None
                            yield '!END'
                            self.rate_limiter.on_success()
                    
                    elif text and get_context and not summary:
                        if smart_history_search:
                            custom_entry = f"Using the current message, with the message history, what is the user is focused on. \nCurrent message: {text}. \n\nPrevious messages: {history}."
                            search_input = self.ai.llm(custom_prompt=custom_entry, model=model, temperature=temperature, timeout=timeout)
                        else:
                            search_input = segment + history if history_search else segment

                        context = self.get_similar(search_input, n=n_context) if not local else self.get_similar_local(search_input, n=n_context)
                        input_ = str(context) if include_context_meta else ''
                        for text in context:
                            input_ += text['data']

                        try:
                            for word in self.ai.llm_w_context_stream(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                full_response += word
                                yield word
                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e

                        if return_context:
                            for item in context:
                                if not metatag:
                                    for tag in item['metadata']:
                                        yield str(item['metadata'][f'{tag}'])
                                else:
                                    if metatag_prefixes:
                                        if metatag_suffixes:
                                            for i in range(len(metatag)):
                                                yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}']) + str(metatag_suffixes[i])
                                        else:
                                            for i in range(len(metatag)):
                                                yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}'])
                                yield item['data']
                            self.update_conversation_history(self.cuid, f'User: {text} \n\nAI: {full_response}') if self.cuid else None
                            yield '!END'
                            self.rate_limiter.on_success()
                        else:
                            self.update_conversation_history(self.cuid, f'User: {text} \n\nAI: {full_response}') if self.cuid else None
                            yield '!END'
                            self.rate_limiter.on_success()

                    else:
                        try:
                            for word in self.ai.llm_stream(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                full_response += word
                                yield word
                            self.rate_limiter.on_success()
                        except Exception as e:
                            raise e
                        self.update_conversation_history(self.cuid, f'User: {text} \n\nAI: {full_response}') if self.cuid else None
                        yield '!END'

                    self.rate_limiter.on_success()
                    break
                except Exception as e:
                    exceptions += 1
                    print(f"API Error: {e}. Applying backoff.")
                    self.rate_limiter.on_failure()
                    if exceptions >= self.rate_limiter.max_attempts:
                        print(f"API Failed too many times, exiting loop: {e}.")
                        break
            
        print("get chat time --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
            

    def print_stream(self, function, printing=True):
        '''
            For local use printing the chat stream. Call 'printing=False' for no pretty printing to be applied
        '''
        full_text= ''
        newlinetime=1
        for word in function:
            if word != '!END' and word:
                full_text += word
                if printing:
                    if len(full_text) / 80 > newlinetime:
                        newlinetime += 1
                        print(f'\n{word}', end='', flush=True)
                    else:
                        print(word, end='', flush=True) 
            else:
                return full_text
        

    def print_vault_data(self, return_data: bool = False):
        ''' 
            Function to print vault data 
        ''' 
        vd = self.cloud_manager.get_mapping()
        # Function to format datetime
        def format_datetime(dt, is_timestamp=False):
            if is_timestamp:
                return datetime.fromtimestamp(dt).strftime("%m/%d/%y, %H:%M")
            else:
                return datetime.fromisoformat(dt).strftime("%m/%d/%y, %H:%M")

        # Formatting and storing data
        formatted_data = []
        for record in vd:
            vr = record['vault']
            total_items = record['total_items']
            last_up = record['last_update']
            if type(last_up) is str:
                last_update = format_datetime(last_up)
            else:
                last_update = format_datetime(last_up, is_timestamp=True)
            last_use = format_datetime(record['last_use'], is_timestamp=True) if 'last_use' in record else ''
            total_use = record.get('total_use', '')  # Default to empty string if 'total_use' is not present
            formatted_data.append([vr, total_items, last_update, last_use, total_use])

        # Calculate the maximum length for the 'vault' field
        max_vault_length = max(len(record[0]) for record in formatted_data)

        if print: # Print formatted data
            header = f"{'| Vault':<{max_vault_length}}   | Total Items | Last Update      | Last Use        | Total Use |"
            print(header)
            print("-" * len(header))
            for row in formatted_data:
                print(f"| {row[0]:<{max_vault_length}} | {row[1]:<11} | {row[2]:<16} | {row[3]:<15} | {row[4]:<8}  |")
        
            return vd if return_data else None
        

    def cloud_stream(self, function):
        '''
            For cloud application yielding the chat stream, like a flask app
        '''
        for word in function:
            yield f"data: {json.dumps({'data': word})} \n\n"


    def update_conversation_history(self, conversation_id, message, metadata_list = None):
        self.reload_vectors()
        try:
            metadata_list = metadata_list if metadata_list else json.loads(self.cloud_manager.download_text_from_cloud(f'{self.vault}/user_history/{conversation_id}/metadata'))
        except:
            metadata_list = []
        message_id = f"{time.time():.0f}"  # Current time
        metadata_list.append({ 'M': message_id, 'L': len(message) })
        t1 = T(target=self.cloud_manager.upload_to_cloud, args=(f'{self.vault}/user_history/{conversation_id}/{message_id}', message))
        t2 = T(target=self.cloud_manager.upload_to_cloud, args=(f'{self.vault}/user_history/{conversation_id}/metadata', json.dumps(metadata_list)))
        t1.start()
        t2.start()
        T(target=call_cloud_save, args=(
            self.user, self.api, self.openai_key, 
            f'{self.vault}/user_history/{conversation_id}/vectors', self.embeddings_model, 
            message, None, 
            message_id, None, None)).start()
        t1.join()
        t2.join()

    def get_conversation_history(self, conversation_id, message):
        history_time = time.time()
        history = ''
        def download_conversation_metadata():
            try:
                return json.loads(self.cloud_manager.download_text_from_cloud(f'{self.vault}/user_history/{conversation_id}/metadata'))
            except:
                return []

        def download_conversation_message(msg_id):
            try:
                return self.cloud_manager.download_text_from_cloud(f'{self.vault}/user_history/{conversation_id}/{msg_id}')
            except:
                return []

        def golden_retriever(metadata_list): 
            one_hour_ago = datetime.now() - timedelta(hours=1)
            return [metadata['M'] for metadata in reversed(metadata_list) if datetime.fromtimestamp(float(metadata['M'])) >= one_hour_ago]

        def vector_search_conversation_history(message):
            return self.get_similar(message, vault=f'{self.vault}/user_history/{conversation_id}/vectors')

        try:
            meta = download_conversation_metadata()
        except:
            meta = None

        message_ids = golden_retriever(meta) if meta != None else []

        print('message_ids:', message_ids) if self.verbose else None

        if message_ids != []:
            history_lines = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_message_id = {executor.submit(download_conversation_message, message_id): message_id for message_id in message_ids}

                # Iterate over the futures as they complete (preserves order)
                for future in as_completed(future_to_message_id):
                    message_content = future.result()
                    history_lines.append(message_content)

            history = '\n'.join(history_lines)
            history = "Recent conversation history:" + history

            vector_similar_results = vector_search_conversation_history(message)
            if vector_similar_results:
                vector_history = []
                now = datetime.now()
                for i in vector_similar_results:
                    message_time = datetime.fromtimestamp(float(i['metadata']['name']))
                    data = i['data']
                    vector_history.append(f'{get_time_statement(now, message_time)} {data}')                

                history = history + "Vector search conversation history:" + '\n'.join(vector_history)

        print('history retrieval took:', time.time() - history_time) if self.verbose else None
        return history


class RateLimiter:
    def __init__(self, max_attempts=30):
        self.base_delay = 1  # Base delay of 1 second
        self.max_delay = 60  # Maximum delay of 60 seconds
        self.backoff_factor = 2
        self.current_delay = self.base_delay
        self.max_attempts = max_attempts

    def on_success(self):
        # Reset delay after a successful call
        self.current_delay = self.base_delay

    def on_failure(self):
        # Apply exponential backoff with a random jitter
        self.current_delay = min(self.max_delay, random.uniform(self.base_delay, self.current_delay * self.backoff_factor))
        time.sleep(self.current_delay)
