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
from threading import Thread as T, Event
from datetime import datetime
from typing import List, Union, Dict
from .ai import openai, OpenAIPlatform, AnthropicPlatform, GrokPlatform, GeminiPlatform, LLMClient, get_all_models
from .cloud_api import (run_flow, run_flow_stream, update_vault_metadata, delete_vault_metadata, update_vault_mapping, 
                        get_vault_mapping, get_user_vault_data, update_user_vault_data, save_custom_prompt, save_personality_message)
from .cloudmanager import CloudManager, VaultStorageManager, as_completed, ThreadPoolExecutor
from .itemize import itemize, name_vecs, get_item, get_vectors, build_return, cloud_name, load_json


class Vault:
    def __init__(self, user: str = None, api_key: str = None, openai_key: str = None, vault: str = None, 
                 embeddings_model: str = None, verbose: bool = False, 
                 model = None, grok_key: str = None, anthropic_key: str = None,
                 gemini_key: str = None, main_prompt = None, main_prompt_with_context = None, personality_message = None):
        ''' 
        >>> Create a vector database instance:
        ```
        vault = Vault(user='your_email',
              api_key='vectorvault_api',
              openai_key='openai_api',
              vault='your_vault_name',
              verbose=True)
        ```

        >>> Add data to the vector database aka "the Vault":
        ```
        vault.add('some text')
        vault.get_vectors()
        vault.save()
        ```

        >>> Plain ChatGPT response:
        `basic_chatgpt_answer = vault.get_chat('some question')`

        >>> RAG response: -> (use 'get_context=True' to read 4 vector similar search results from the database before responding])
        `rag_answer = vault.get_chat('some question', get_context=True)`

        >>> Change the model with the `model` param in get_chat:
        `gpt4_rag_answer = vault.get_chat('some question', get_context=True, model='gpt-4')`

        '''
        self.user = user.lower()
        self.vault = vault.strip() if vault else 'home'
        self.api = api_key
        self.verbose = verbose
        self.embeddings_model = embeddings_model if embeddings_model else 'text-embedding-3-small'
        self.dims = 1536 if embeddings_model != 'text-embedding-3-large' else 3072
        
        self._cloud_manager = None
        self._cloud_manager_thread = None
        self._cloud_manager_ready = Event()

        def _init_cloud_manager():
            cm = CloudManager(self.user, self.api, self.vault)
            self._cloud_manager = cm
            if cm.item_mapping and not self.map:
                self.map = dict(cm.item_mapping)
            self._cached_prompt_with_context = cm.custom_prompt_with_context
            self._cached_prompt_no_context = cm.custom_prompt_no_context
            self._cached_personality = cm.personality_message
            self._cloud_manager_ready.set()

        if user and api_key:
            self._cloud_manager_thread = T(target=_init_cloud_manager, daemon=True)
            self._cloud_manager_thread.start()
        else:
            self._cloud_manager_ready.set()
        
        # Lazy initialization for other components
        self._storage = None
        self._vectors = None
        self._all_models = get_all_models()
        self._platforms_initialized = False
        
        # Initialize state variables
        self.x = 0
        self.x_checked = False
        self.x_loaded_checked = False
        self.vecs_loaded = False
        self.load_json = load_json
        if self._cloud_manager and self._cloud_manager.item_mapping:
            self.map = self._cloud_manager.item_mapping
        else:
            self.map = {}
        self.items = []
        self.item_cache = {}  # Cache for fetched items: {uuid: {'data': text, 'metadata': meta}}
        self.last_time = None
        self.saved_already = False
        self._ai = None
        self.ai_loaded = False
        self.rate_limiter = RateLimiter(max_attempts=30)
        self.model = model # your chosen defualt model
        
        # Store API keys for lazy initialization
        self.openai_key = openai_key
        self.grok_key = grok_key
        self.anthropic_key = anthropic_key
        self.gemini_key = gemini_key
        
        # Set OpenAI API key if provided
        if openai_key:
            openai.api_key = openai_key
        
        # Store platform instances for lazy initialization
        self._openai = None
        self._grok = None
        self._anthropic = None
        self._gemini = None
        
        # Configuration settings
        self.fine_tuned_context_window = 128000
        self.main_prompt = main_prompt if main_prompt else "{content}"
        self.main_prompt_with_context = main_prompt_with_context if main_prompt_with_context else """Use the following Context to answer the Question at the end.
    Answer as if you were the modern voice of the context, without referencing the context or mentioning
    the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

    Additional Context: {context}

    Question: {content}
    """    
        self.personality_message = personality_message if personality_message else ""
        
        # Initialize current vault mapping from CloudManager if available
        self.map = {}
        self._cached_prompt_with_context = self.main_prompt_with_context
        self._cached_prompt_no_context = self.main_prompt
        self._cached_personality = self.personality_message

    @property
    def cloud_manager(self):
        """Get cloud manager (initialized eagerly in __init__)"""
        self._finalize_cloud_manager_initialization()
        if self._cloud_manager is None:
            raise ValueError("User and API key are required for cloud operations")
        return self._cloud_manager

    def _finalize_cloud_manager_initialization(self):
        if self._cloud_manager_ready.is_set():
            return

        if hasattr(self, "_cloud_manager_thread") and self._cloud_manager_thread:
            self._cloud_manager_thread.join()
            self._cloud_manager_thread = None

            if self._cloud_manager:
                if self._cloud_manager.item_mapping and not self.map:
                    self.map = dict(self._cloud_manager.item_mapping)

                # Hydrate cached prompts/personality from CloudManager init data
                if self._cloud_manager.custom_prompt_with_context:
                    self._cached_prompt_with_context = self._cloud_manager.custom_prompt_with_context
                if self._cloud_manager.custom_prompt_no_context:
                    self._cached_prompt_no_context = self._cloud_manager.custom_prompt_no_context
                if self._cloud_manager.personality_message:
                    self._cached_personality = self._cloud_manager.personality_message

                self._apply_cached_prompts_to_ai()

        if not self._cloud_manager_ready.is_set():
            self._cloud_manager_ready.set()

    @property
    def storage(self):
        """Lazy initialization of storage manager"""
        if self._storage is None:
            self._storage = VaultStorageManager(self.vault, self.cloud_manager)
        return self._storage

    @property
    def vectors(self):
        """Lazy initialization of vectors"""
        if self._vectors is None:
            self._vectors = get_vectors(self.dims)
        return self._vectors

    @vectors.setter
    def vectors(self, value):
        """Allow setting vectors directly"""
        self._vectors = value

    @property
    def all_models(self):
        """Return cached model map"""
        return self._all_models

    def _initialize_platforms(self):
        """Initialize AI platforms only when needed"""
        if not self._platforms_initialized:
            self._openai = OpenAIPlatform()
            self._grok = GrokPlatform(self.grok_key)
            self._anthropic = AnthropicPlatform(self.anthropic_key)
            self._gemini = GeminiPlatform(self.gemini_key)
            self._platforms_initialized = True

    @property
    def openai(self):
        """Lazy initialization of OpenAI platform"""
        if self._openai is None:
            self._openai = OpenAIPlatform()
        return self._openai

    @property
    def grok(self):
        """Lazy initialization of Grok platform"""
        if self._grok is None:
            self._grok = GrokPlatform(self.grok_key)
        return self._grok

    @property
    def anthropic(self):
        """Lazy initialization of Anthropic platform"""
        if self._anthropic is None:
            self._anthropic = AnthropicPlatform(self.anthropic_key)
        return self._anthropic

    @property
    def gemini(self):
        """Lazy initialization of Gemini platform"""
        if self._gemini is None:
            self._gemini = GeminiPlatform(self.gemini_key)
        return self._gemini

    def get_total_items(self, vault: str = None):
        '''
            Returns the total number of vectored items in the Vault
        '''
        target_vault = vault if vault else self.vault
        mapping = self.load_mapping(target_vault)
        if mapping:
            return len(mapping)
        return 0

    def get_total_vectors(self):
        '''
            Returns the total number of vectors in the Vault
        '''
        self.check_index_loaded()
        return self.vectors.get_n_items()


    def get_tokens(self, text: str):
        '''
            Returns the number of tokens for any given text
        '''
        self.load_ai()
        return self._ai.get_tokens(text)
    

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
        

    def add_fine_tuned_model_to_platform(self, model_name: str, platform='openai', token_limit=None):
        """
        Adds a fine-tuned model to the model_token_limits dictionary of the specified platform.

        Args:
            model_name (str): The name of the fine-tuned model.
            platform (str): The target platform ('openai', 'grok', 'anthropic', 'gemini').
            token_limit (int): The token limit for the fine-tuned model.
        """
        if platform == 'openai':
            platform_instance = self.openai
        elif platform == 'grok':
            platform_instance = self.grok
        elif platform == 'anthropic':
            platform_instance = self.anthropic
        elif platform == 'gemini':
            platform_instance = self.gemini
        else:
            raise ValueError(f"Unknown platform: {platform}")
        
        if token_limit:
            self.fine_tuned_context_window = token_limit

        token_limit = token_limit if token_limit else 128000
        platform_instance.model_token_limits[model_name] = token_limit 
        print(f"Fine-tuned model '{model_name}' added to {platform} with token limit {token_limit}.") if self.verbose else 0


    def get_client_from_model(self, model: str):
        """
        Returns the correct platform instance to use for a given model.

        Args:
            model (str): The name of the model.

        Returns:
            LLMPlatform: The appropriate platform instance.
        """
        client_kwargs = {
            "fine_tuned_context_window": self.fine_tuned_context_window,
            "main_prompt": self.main_prompt,
            "main_prompt_with_context": self.main_prompt_with_context,
            "personality_message": self.personality_message
        }

        # Handle 'default' as a special case - always use OpenAI's default
        if model == 'default':
            return LLMClient(self.openai, **client_kwargs)

        # Prefer checking OpenAI first to avoid initializing other SDKs unnecessarily
        if model in self.openai.model_token_limits:
            return LLMClient(self.openai, **client_kwargs)
        elif model in self.grok.model_token_limits:
            return LLMClient(self.grok, **client_kwargs)
        elif model in self.gemini.model_token_limits:
            return LLMClient(self.gemini, **client_kwargs)
        elif model in self.anthropic.model_token_limits:
            return LLMClient(self.anthropic, **client_kwargs)
        else:
            print(f"Model '{model}' not found in any platform -> adding as a fine-tuned OpenAI model...")
            self.add_fine_tuned_model_to_platform(model)
            return LLMClient(self.openai, **client_kwargs)


    def _get_prompt_from_cache(self, context=True):
        if context:
            return self._cached_prompt_with_context or self.main_prompt_with_context
        return self._cached_prompt_no_context or self.main_prompt

    def _get_personality_from_cache(self):
        return self._cached_personality or self.personality_message

    def _apply_cached_prompts_to_ai(self):
        if not self._ai:
            return
        self._ai.main_prompt_with_context = self._get_prompt_from_cache(context=True)
        self._ai.main_prompt = self._get_prompt_from_cache(context=False)
        self._ai.personality_message = self._get_personality_from_cache()
        self._ai.set_prompts()

    def _ensure_prompts_ready(self):
        self._finalize_cloud_manager_initialization()

    @property
    def ai(self):
        self.load_ai() 
        return self._ai
            

    def load_ai(self, model: str = None) -> None:
        '''
            Loads the AI functions - internal function
        '''

        target_model = model or self.all_models.get('default') or 'default'

        if self._ai is None or model:
            self._ai = self.get_client_from_model(model=target_model)

        self.ai_loaded = True
        self._apply_cached_prompts_to_ai()
            

    def save_personality_message(self, text: str):
        '''
            Saves personality_message to the vault and use it by default from now on
        '''
        self._cached_personality = text
        self._apply_cached_prompts_to_ai()
        self.cloud_manager.upload_personality_message(text)
        print(f"Personality message saved") if self.verbose else 0
            

    def fetch_personality_message(self):
        '''
            Retrieves personality_message from the vault if it is there or else use the defualt
        '''
        if self._cached_personality is not None:
            return self._cached_personality
        if not self._cloud_manager_ready.is_set():
            return self.personality_message
        try:
            personality_message = self.cloud_manager.download_personality_message()
            if personality_message is not None:
                self._cached_personality = personality_message
                return personality_message
        except:
            pass
        
        if self.ai_loaded:
            personality_message = self._ai.personality_message
        else: # only when called externally in some situations
            personality_message = self.personality_message
                
        return personality_message
    

    def save_custom_prompt(self, text: str, context=True):
        '''
            Saves custom_prompt to the vault and use it by default from now on
            Param: "context" True = context prompt ; False = main prompt
        '''
        if context:
            self._cached_prompt_with_context = text
        else:
            self._cached_prompt_no_context = text
        self._apply_cached_prompts_to_ai()
        self.cloud_manager.upload_custom_prompt(text, context)
        print(f"Custom prompt saved") if self.verbose else 0
            

    def fetch_custom_prompt(self, context=True):
        '''
            Retrieves custom_prompt from the vault if there or eles use defualt - (used for get_context = True responses)
            context == False will return custom prompt for not context situations
        '''
        cache = self._cached_prompt_with_context if context else self._cached_prompt_no_context
        if cache is not None:
            return cache
        if not self._cloud_manager_ready.is_set():
            return self.main_prompt_with_context if context else self.main_prompt
        try:
            prompt = self.cloud_manager.download_custom_prompt(context)
            if prompt is not None:
                if context:
                    self._cached_prompt_with_context = prompt
                else:
                    self._cached_prompt_no_context = prompt
                return prompt
        except:
            pass
        
        if self.ai_loaded:
            prompt = self._ai.main_prompt_with_context if context else self._ai.prompt
        else: # only when called externally in some situations
            self.load_ai()
            prompt = self.main_prompt_with_context if context else self.main_prompt
            
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
            futures = []
            for item in self.items:
                item_text, item_id, item_meta = get_item(item)
                item_uuid = self.map.get(str(item_id))
                future = executor.submit(self.cloud_manager.upload, item_uuid, item_text, item_meta, vault)
                futures.append(future)
                self.item_cache[item_uuid] = build_return(item_text, item_meta)
                total_saved_items += 1
            
            # Wait for all uploads to complete
            for future in futures:
                future.result()

        all_vaults = self.cloud_manager.download_vaults_list_from_cloud()

        if self.vault not in all_vaults and all_vaults:
            all_vaults.append(self.vault)
            self.cloud_manager.upload_vaults_list(all_vaults)

        self.upload_vectors(vault)
        self.update_vault_data(this_vault_only=True)
        self.cloud_manager.build_data_update()
        print(f"upload time --- {(time.time() - start_time)} seconds --- {total_saved_items} items saved") if self.verbose else 0
            

    def create_vault(self, vault_name: str = None):
        '''
        Creates and registers a new vault without requiring data to be added first.
        The vault will be properly listed in vault mappings and vault lists.
        
        Args:
            vault_name (str, optional): Name of the vault to create. 
                                      If None, uses the current vault name.
        '''
        vault_name = vault_name if vault_name else self.vault
        start_time = time.time()
        
        # Check if vault already exists
        existing_vaults = self.cloud_manager.download_vaults_list_from_cloud()
        if vault_name in existing_vaults:
            print(f"Vault '{vault_name}' already exists") if self.verbose else 0
            return
        
        # Create empty mapping via API
        update_vault_mapping(self.user, self.api, vault_name, {})
        
        # Create empty vectors file in GCS
        empty_vectors = get_vectors(self.dims)
        empty_vectors.build(10)  # Build the index before saving (even if empty)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            vector_temp_file_path = temp_file.name
            empty_vectors.save(vector_temp_file_path)
            byte = os.path.getsize(vector_temp_file_path)
            self.cloud_manager.upload_temp_file(vector_temp_file_path, name_vecs(vault_name, self.user, self.api, byte))
        
        self.delete_temp_file(vector_temp_file_path)
        
        # Add vault to vaults_list via API (and invalidate cached init data)
        existing_vaults.append(vault_name)
        # Use CloudManager helper so its cache stays consistent
        self.cloud_manager.upload_vaults_list(existing_vaults)
        
        # Create initial vault metadata via API
        now = time.time()
        update_vault_metadata(
            self.user,
            self.api,
            vault_name,
            total_items=0,
            last_update=now,
            last_use=now,
            total_use=1
        )
        
        # Trigger cloud data update for this user/vault
        self.cloud_manager.build_data_update()
        
        # Copy prompts and personality from the current vault into the new one
        save_personality_message(self.user, self.api, vault_name, self.fetch_personality_message())
        save_custom_prompt(self.user, self.api, vault_name, self.fetch_custom_prompt(context=False), context=False)
        save_custom_prompt(self.user, self.api, vault_name, self.fetch_custom_prompt(context=True), context=True)
        
        print(f"Vault '{vault_name}' created successfully --- {(time.time() - start_time):.2f} seconds") if self.verbose else 0


    def clear_cache(self):
        '''
            Clears the cache for all the loaded items 
        '''
        self.reload_vectors()
        self.vecs_loaded = True
        self.saved_already = False


    def duplicate_vault(self, new_vault_name, embeddings_model=None):
        vault = Vault(user=self.user, api_key=self.api, ai_key=self.ai_key, vault=self.vault, verbose=self.verbose, chat_ai=self.chat_ai)
        vault2 = Vault(user=self.user, api_key=self.api, ai_key=self.ai_key, vault=new_vault_name, verbose=self.verbose, chat_ai=self.chat_ai)

        for i in range(vault.get_total_items()):
            item = vault.get_items([i])[0]['data']
            vault2.add(item)
            print(f"\r{i}", end='') if self.verbose else None

        vault2.get_vectors()
        vault2.save()

        self.load_ai()
        vault2.save_custom_prompt(self._ai.main_prompt, False)
        vault2.save_custom_prompt(self._ai.main_prompt_with_context)
        vault2.save_personality_message(self._ai.personality_message)


    def delete(self):
        '''
            Deletes the entire Vault and all contents
        '''
        print('Deleting started. Note: this can take a while for large datasets') if self.verbose else 0
            
        # Clear the local vector data
        self._vectors = get_vectors(self.dims)
        self.items.clear()
        self.item_cache.clear()  
        self.cloud_manager.delete()
        self.x = 0
        vaults = self.cloud_manager.download_vaults_list_from_cloud()
        try:
            vaults.remove(self.vault)
        except: 
            pass
        self.cloud_manager.upload_vaults_list(vaults)
        self.update_vault_data(remove_vault=True)
        self.map = {}
        print(f'Vault: "{self.vault}" deleted')
    

    def get_vaults(self, vault: str = None) -> list:
        '''
            Returns a list of vaults within the current vault directory 
        '''
        vault = self.vault if vault is None else vault

        if vault == '':
            all_vaults = self.get_all_vaults()
            if all_vaults == []:
                all_vaults = self.cloud_manager.list_vaults(vault)
                self.cloud_manager.upload_vaults_list(all_vaults)
                return all_vaults
            else:
                return all_vaults
        
        return self.cloud_manager.list_vaults(vault)
    

    def get_all_vaults(self) -> list:
        '''
            Returns a list of vaults within the current vault directory 
        '''
        try: 
            return self.cloud_manager.download_vaults_list_from_cloud()
        except:
            return []
    

    def list_cloud_vaults(self):
        '''
            Returns a list of all the cloud vaults 
        '''
        return self.cloud_manager.list_vaults('')
    

    def save_mapping(self, vault = None):
        vault = self.vault if not vault else vault
        
        # Update CloudManager cache if saving current vault
        if vault == self.vault:
            self.cloud_manager.item_mapping = self.map.copy()
        
        # Save to backend API
        update_vault_mapping(self.user, self.api, vault, self.map)
        
        # Invalidate cache so it gets reloaded with latest data
        if vault == self.vault:
            self.cloud_manager._init_data_loaded = False
            self.cloud_manager._init_data = None
        

    def remap(self, item_id):
        for i in range(item_id, len(self.map) - 1):
            self.map[str(i)] = self.map[str(i + 1)]
        self.map.popitem()


    def remap_item_numbers(self):
        new_map = {}
        new_item_number = 0
        self.load_mapping()
        
        # Iterate through all items in the current map
        for old_key, uuid in self.map.items():
            meta_path = self.cloud_manager.cloud_name(self.vault, uuid, self.user, self.api, meta=True)
            if self.cloud_manager.item_exists(uuid):
                meta_data = json.loads(self.cloud_manager.download_text_from_cloud(meta_path))
                meta_data['item_id'] = new_item_number
                self.cloud_manager.upload_to_cloud(meta_path, json.dumps(meta_data))
                new_map[str(new_item_number)] = uuid
                new_item_number += 1
        
        self.map = new_map
        self.save_mapping()
        self.update_vault_data(this_vault_only=True)
        
        print(f"Item numbering fixed. Total items: {new_item_number}")

        
    def update_vault_data(self, this_vault_only=False, remove_vault=False):
        """Update the master vault data file that tracks all vaults"""
        nary = []
        try:
            # Get current vault data from API
            vault_data = get_user_vault_data(self.user, self.api)
            if not vault_data:
                vault_data = []
            
            all_vaults = [i['vault'] for i in vault_data]
            
            # Add current vault if it doesn't exist
            if self.vault not in all_vaults and not remove_vault:
                nary.append({
                    'vault': self.vault,
                    'total_items': self.get_total_items(),
                    'last_update': time.time(),
                    'last_use': time.time(),
                    'total_use': 1
                })
            
            # Process existing vaults
            for i in vault_data:
                if not this_vault_only and not remove_vault:
                    total_items = self.get_total_items(i['vault'])
                else:
                    total_items = i['total_items']
                
                if this_vault_only and i['vault'] == self.vault:
                    total_items = self.get_total_items()
                
                vault_dict = {
                    'vault': i['vault'],
                    'total_items': total_items,
                    'last_update': i.get('last_update', time.time())
                }
                
                if 'last_use' in i:
                    vault_dict['last_use'] = i['last_use']
                if 'total_use' in i:
                    vault_dict['total_use'] = i['total_use']
                
                # Update timestamps for current vault
                if i['vault'] == self.vault and not remove_vault:
                    vault_dict['last_update'] = time.time()
                
                if remove_vault and self.vault == i['vault']:
                    pass  # Skip this vault
                else:
                    nary.append(vault_dict)
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            # Create default entry if all fails
            nary.append({
                'vault': 'demo',
                'total_items': 0,
                'last_update': time.time(),
                'last_use': time.time(),
                'total_use': 1
            })
        
        # Save to API
        update_user_vault_data(self.user, self.api, nary)
        
        # Also update individual vault metadata in backend if needed
        if not remove_vault:
            update_vault_metadata(
                self.user,
                self.api,
                self.vault,
                last_update=time.time()
            )
        else:
            delete_vault_metadata(self.user, self.api, self.vault)
        
        # Update cached metadata on CloudManager
        if remove_vault:
            self.cloud_manager.vault_metadata = {}
        else:
            for entry in nary:
                if entry['vault'] == self.vault:
                    self.cloud_manager.vault_metadata = {
                        k: v for k, v in entry.items()
                        if k in ('last_update', 'last_use', 'total_items', 'total_use')
                    }
                    break
        
        # Invalidate cache
        self.cloud_manager._init_data_loaded = False
        self.cloud_manager._init_data = None


    def hard_remap_vault_data(self):
        """Rebuild user vault metadata using backend APIs (no GCS)."""
        try:
            existing = get_user_vault_data(self.user, self.api) or []
        except Exception:
            existing = []
        existing_map = {}
        for entry in existing:
            if isinstance(entry, dict) and 'vault' in entry:
                existing_map[entry['vault']] = entry

        rebuilt_entries = []
        for vault_name in self.get_vaults(''):
            mapping = get_vault_mapping(self.user, self.api, vault_name) or {}
            if isinstance(mapping, list):
                mapping = {str(index): value for index, value in enumerate(mapping)}
            total_items = len(mapping)

            prior = existing_map.get(vault_name, {})
            vault_dict = {
                'vault': vault_name,
                'total_items': total_items,
                'last_update': prior.get('last_update', time.time()),
                'last_use': prior.get('last_use', time.time()),
                'total_use': prior.get('total_use', 0)
            }
            rebuilt_entries.append(vault_dict)

        update_user_vault_data(self.user, self.api, rebuilt_entries)

        # Refresh CloudManager cached metadata for current vault
        current_meta = next((entry for entry in rebuilt_entries if entry['vault'] == self.vault), {})
        self.cloud_manager.vault_metadata = current_meta
        self.cloud_manager._init_data_loaded = False
        self.cloud_manager._init_data = None


    def delete_items(self, item_ids: List[int], trees: int = 10) -> None:
        '''
        Deletes one or more items from item_id(s) passed in.
        item_ids is an int or a list of integers
        '''
        def download_and_upload(old_id, new_id):
            metadata = json.loads(self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.old_map[str(old_id)], self.user, self.api, meta=True)))
            metadata['item_id'] = new_id
            self.cloud_manager.upload_to_cloud(cloud_name(self.vault, self.old_map[str(old_id)], self.user, self.api, meta=True), json.dumps(metadata))
            self.map[str(new_id)] = self.old_map[str(old_id)]

        def rebuild_vectors(item_ids):
            '''Deletes target vectors and rebuilds the database'''
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)
            new_id = 0

            with ThreadPoolExecutor() as executor:
                futures = []

                for old_id in range(num_existing_items):
                    if old_id not in item_ids:
                        vector = self.vectors.get_item_vector(old_id)
                        new_index.add_item(new_id, vector)
                        time.sleep(.05)
                        futures.append(executor.submit(download_and_upload, old_id, new_id))
                        new_id += 1

                # Ensure all threads complete
                for future in as_completed(futures):
                    future.result()

            self.vectors = new_index
            self.vectors.build(trees)
            self.upload_vectors()

        item_ids = [item_ids] if isinstance(item_ids, int) else item_ids
        self.load_vectors()
        self.old_map = self.map.copy()

        for item_id in item_ids:
            # Remove deleted items from cache
            item_uuid = self.old_map.get(str(item_id))
            if item_uuid and item_uuid in self.item_cache:
                del self.item_cache[item_uuid]
            
            self.cloud_manager.delete_item(item_uuid)
            self.remap(item_id)

        rebuild_vectors(item_ids)
        self.update_vault_data(this_vault_only=True)
        self.cloud_manager.build_data_update()
        print(f'Items {item_ids} deleted and database rebuilt') if self.verbose else 0


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
        item_uuid = self.map.get(str(item_id))
        
        # Get existing metadata to preserve it
        meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, item_uuid, self.user, self.api, meta=True))
        metadata = json.loads(meta_data)
        
        # Upload new text
        self.cloud_manager.upload_to_cloud(cloud_name(self.vault, item_uuid, self.user, self.api, item=True), new_text)
        
        # Update cache with new text and existing metadata
        if item_uuid:
            self.item_cache[item_uuid] = build_return(new_text, metadata)
        
        edit_vector(item_id, self.process_batch([new_text], never_stop=False, loop_timeout=180)[0])

        self.update_vault_data(this_vault_only=True)
        self.cloud_manager.build_data_update()
        print(f'Item {item_id} edited') if self.verbose else 0


    def edit_item_meta(self, item_id: int, metadata) -> None:
        '''
            Edit and save any item's metadata
        '''
        item_uuid = self.map.get(str(item_id))
        
        # Update cache with new metadata if item is cached
        if item_uuid and item_uuid in self.item_cache:
            # Get existing data from cache and update metadata
            cached_item = self.item_cache[item_uuid]
            self.item_cache[item_uuid] = build_return(cached_item['data'], metadata)
        
        self.cloud_manager.upload_to_cloud(cloud_name(self.vault, item_uuid, self.user, self.api, meta=True), json.dumps(metadata))
        print(f'Item {item_id} metadata saved') if self.verbose else 0


    def clear_cache(self):
        '''
            Manually clear the item cache. 
            The cache is automatically updated when saving, deleting, or editing items.
            Use this if you need to force a full refresh of all cached data.
        '''
        cache_size = len(self.item_cache)
        self.item_cache.clear()
        print(f'Item cache cleared ({cache_size} items removed)') if self.verbose else 0


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
                if not self.vecs_loaded:
                    self.load_vectors()
                self.reload_vectors()
            self.x_checked = True
            print("initialize index --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0

    def load_mapping(self, vault = None):
        self._finalize_cloud_manager_initialization()
        target_vault = vault if vault else self.vault 
        return self._get_mapping_for_vault(target_vault)

    def _get_mapping_for_vault(self, vault):
        if vault == self.vault:
            if not self.map:
                if not self._cloud_manager:
                    raise ValueError("Cloud manager not initialized")
                self.map = self._cloud_manager.item_mapping
            return self.map
        return get_vault_mapping(self.user, self.api, vault)

    def _download_vectors_for_vault(self, vault):
        temp_file_path = self.cloud_manager.download_to_temp_file(name_vecs(vault, self.user, self.api))
        vectors = get_vectors(self.dims)
        vectors.load(temp_file_path)
        self.delete_temp_file(temp_file_path)
        return vectors

    def add_to_map(self):
        self.map[str(self.x)] = str(uuid.uuid4())
        self.cloud_manager.item_mapping = self.map
        self.x +=1
    
    def delete_temp_file(self, temp_file_path, attempts = 5):
        for i in range(attempts):
            try:
                os.delete(temp_file_path)
                break
            except: 
                time.sleep(.01)

    def load_vectors(self, vault = None):
        start_time = time.time()
        target_vault = vault if vault else self.vault 

        if target_vault != self.vault:
            return self._download_vectors_for_vault(target_vault)

        t = T(target=self.load_mapping, args=(self.vault,))
        t.start()
        temp_file_path = self.cloud_manager.download_to_temp_file(name_vecs(self.vault, self.user, self.api))
        self._vectors = get_vectors(self.dims)
        self._vectors.load(temp_file_path)
        self.delete_temp_file(temp_file_path)
        t.join()
        self.vecs_loaded = True
        self.x_checked = False
        print("get load vectors --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
        return self._vectors

    def reload_vectors(self):
        num_existing_items = self.vectors.get_n_items()
        new_index = get_vectors(self.dims)
        count = -1
        for i in range(num_existing_items):
            count += 1
            vector = self.vectors.get_item_vector(i)
            new_index.add_item(i, vector)
        self.x = count + 1
        self._vectors = new_index
        self.vecs_loaded = False


    def upload_vectors(self, vault = None):
        vault = vault if vault else self.vault
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            vector_temp_file_path = temp_file.name
            self.vectors.save(vector_temp_file_path)
            byte = os.path.getsize(vector_temp_file_path)
            self.cloud_manager.upload_temp_file(vector_temp_file_path, name_vecs(vault, self.user, self.api, byte))
        
        self.delete_temp_file(vector_temp_file_path)
        self.save_mapping(vault)
        self.items.clear()
        # Don't clear cache - it's already populated/updated by save(), edit_item(), etc.
        self._vectors = get_vectors(self.dims)
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
    

    def _fetch_and_cache_item(self, item_uuid: str, vault_name: str, distance: float = None):
        '''
            Internal helper method to fetch an item from cloud and cache it.
            Returns the item data with optional distance.
        '''
        # Download item and metadata from cloud
        item_data = self.cloud_manager.download_text_from_cloud(
            cloud_name(vault_name, item_uuid, self.user, self.api, item=True)
        )
        meta_data = self.cloud_manager.download_text_from_cloud(
            cloud_name(vault_name, item_uuid, self.user, self.api, meta=True)
        )
        meta = json.loads(meta_data)
        
        # Build and cache the result (without distance for reusability)
        result_no_distance = build_return(item_data, meta)
        self.item_cache[item_uuid] = result_no_distance
        
        # Return with distance if provided
        if distance is not None:
            return build_return(item_data, meta, distance)
        return result_no_distance


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
        target_vault = vault if vault else self.vault
        _map = self.load_mapping(target_vault)
        
        # Separate cached and uncached items
        items_to_fetch = []  # List of (index, item_id, uuid) tuples
        cache_hits = 0
        
        for i, item_id in enumerate(ids):
            item_uuid = _map.get(str(item_id))
            if item_uuid is None:
                results[i] = None
                continue
                
            # Check cache first
            if item_uuid in self.item_cache:
                results[i] = self.item_cache[item_uuid]
                cache_hits += 1
            else:
                items_to_fetch.append((i, item_id, item_uuid))
        
        # Fetch uncached items from cloud
        if items_to_fetch:
            def fetch_item(index, item_id, item_uuid):
                result = self._fetch_and_cache_item(item_uuid, target_vault)
                return index, result

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(fetch_item, idx, item_id, uuid) for idx, item_id, uuid in items_to_fetch]
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result  # Insert each result at its corresponding index
        
        if self.verbose:
            print(f"Retrieved {len(ids)} items ({cache_hits} from cache, {len(items_to_fetch)} from cloud) --- %s seconds ---" % (time.time() - start_time))
        
        return results

    
    def get_items_by_vector(self, vector: list, n: int = 4, include_distances: bool = False, vault = None):
        '''
            Internal function that returns vector similar items. Requires input vector, returns similar items
        '''
        try:
            target_vault = vault if vault else self.vault
            mapping = self.load_mapping(target_vault)
            vectors = self.load_vectors(target_vault)
            start_time = time.time()
            if not include_distances:
                vecs = vectors.get_nns_by_vector(vector, n)
                results = [None] * len(vecs)  # Pre-fill the results list with placeholders
                items_to_fetch = []  # List of (index, vec, uuid) tuples
                cache_hits = 0

                # Check cache first
                for i, vec in enumerate(vecs):
                    item_uuid = mapping.get(str(vec))
                    if item_uuid is None:
                        results[i] = None
                        continue
                    
                    # Check cache
                    if item_uuid in self.item_cache:
                        results[i] = self.item_cache[item_uuid]
                        cache_hits += 1
                    else:
                        items_to_fetch.append((i, vec, item_uuid))

                # Fetch uncached items
                if items_to_fetch:
                    def fetch_item(index, vec, item_uuid):
                        result = self._fetch_and_cache_item(item_uuid, target_vault)
                        return index, result
                    
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch_item, idx, vec, uuid) for idx, vec, uuid in items_to_fetch]
                        for future in as_completed(futures):
                            index, result = future.result()
                            results[index] = result  # Insert each result at its corresponding index

                if self.verbose:
                    print(f"get {n} items back ({cache_hits} from cache, {len(items_to_fetch)} from cloud) --- %s seconds ---" % (time.time() - start_time))
                return results
            else:
                vecs, distances = vectors.get_nns_by_vector(vector, n, include_distances=include_distances)
                results = [None] * len(vecs)  # Pre-fill the results list with placeholders
                items_to_fetch = []  # List of (index, vec, uuid, distance) tuples
                cache_hits = 0

                # Check cache first
                for i, (vec, distance) in enumerate(zip(vecs, distances)):
                    item_uuid = mapping.get(str(vec))
                    if item_uuid is None:
                        results[i] = None
                        continue
                    
                    # Check cache - we need to add distance even if cached
                    if item_uuid in self.item_cache:
                        cached_item = self.item_cache[item_uuid]
                        # Build result with distance
                        result = build_return(cached_item['data'], cached_item['metadata'], distance)
                        results[i] = result
                        cache_hits += 1
                    else:
                        items_to_fetch.append((i, vec, item_uuid, distance))

                # Fetch uncached items
                if items_to_fetch:
                    def fetch_item(index, vec, item_uuid, distance):
                        result = self._fetch_and_cache_item(item_uuid, target_vault, distance)
                        return index, result

                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch_item, idx, vec, uuid, dist) for idx, vec, uuid, dist in items_to_fetch]
                        for future in as_completed(futures):
                            index, result = future.result()
                            results[index] = result  # Insert each result at its corresponding index

                if self.verbose:
                    print(f"get {n} items back ({cache_hits} from cache, {len(items_to_fetch)} from cloud) --- %s seconds ---" % (time.time() - start_time))
                return results
        except:
            return [{'data': 'No data has been added', 'metadata': {'no meta': 'No metadata has been added'}}]
    
    
    def download_database_to_json(self, return_meta = False) -> dict:
        '''
        Download all items from the database to json.
        Returns a dictionary with the sequential number of the item (item number) as the key,
        and a dictionary containing the item data and metadata as the value.

        - Example Usage:
            all_items = vault.download_database_to_json()

        Sample return:
        {
            1: {
                'data': 'The Project Gutenberg eBook of The Prince...',
            },
            2: {
                'data': 'Another item content...',
            },
            ...
        }
        '''
        start_time = time.time()
        
        def fetch_item(item_id):
            # Function to fetch a single item, to be run in parallel
            item_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, item=True))
            meta_data = self.cloud_manager.download_text_from_cloud(cloud_name(self.vault, self.map[str(item_id)], self.user, self.api, meta=True))
            meta = json.loads(meta_data) 
            if return_meta:
                return item_id, {
                    'data': item_data,
                    'metadata': meta
                } 
            else:
                return item_id, {
                    'data': item_data,
                }

        self.load_vectors() # also loads mapping

        results = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_item, item_id) for item_id in self.map.keys()]
            for future in as_completed(futures):
                item_number, result = future.result()
                results[item_number] = result

        results = {k: results[k] for k in sorted(results.keys())}

        print(f"Retrieved {len(results)} items --- {time.time() - start_time:.2f} seconds ---") if self.verbose else None
        
        return json.dumps(results)
    
    
    def upload_database_from_json(self, json_data: Union[str, dict]):
        """
        Replace your entire vault from a JSON string or dictionary.
        The JSON should be in the format output by get_all_items().

        :param json_data: A JSON string or dictionary containing items to be added.
        Each item should have 'data' and 'metadata' fields.

        Usage:
        # use with caution
        vault.upload_database_from_json(json_string_or_dict)

        """
        def parse_json(string):
            try:
                return json.loads(string)
            except json.JSONDecodeError:
                print("Warning: JSON decoding failed. Attempting to clean and retry...") if self.verbose else 0
                # Attempt to clean: strip outer quotes if present
                cleaned = string.strip().strip('"')
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    print("Error: Failed to parse JSON after cleaning attempt. Skipping upload.") if self.verbose else 0
                    return {}  # Return empty dict to adapt and continue without raising

        # Load the JSON data with robust parsing
        if not isinstance(json_data, str):
            items_dict = json_data  # Already a dict, use as is
        else:
            # Initial parse for string input
            result = parse_json(json_data)
            
            # If the result is still a string (double-wrapped), parse again
            if isinstance(result, str):
                result = parse_json(result)
            
            # After parsing, ensure it's a dict
            if not isinstance(result, dict):
                print("Warning: Parsed result is not a dictionary. Using empty dict to continue.") if self.verbose else 0
                items_dict = {}
            else:
                items_dict = result

        print('Loaded json items') if self.verbose else 0
        print('Deleting cloud items') if self.verbose else 0

        self.delete_items([i for i in range(self.get_total_items())])

        items_added = 0
        for item_number, item_data in sorted(items_dict.items()):
            if 'data' not in item_data:
                print(f"Skipping item {item_number}: Invalid format") if self.verbose else 0
                continue

            self.add_item(text=item_data['data'], meta=item_data.get('metadata', None))
            items_added += 1
        
        print('Cloud items deleted') if self.verbose else 0
        self.get_vectors()
        print('Saving new json items to the cloud') if self.verbose else 0
        self.save()

        print(f"Successfully saved {items_added} items to the vault.") if self.verbose else 0
        

    def get_similar(self, text: str, n: int = 4, include_distances: bool = False, vault = None):
        '''
            Returns similar items from the Vault as the text you enter.

            Param `include_distances = True` adds the "distance" field to the return.
            The distance can be useful for assessing similarity differences in the items returned. 
            Each item has its' own distance number, and this changes the structure of the output.
        '''
        T(target=self.cloud_manager.build_update, args=(n,)).start()
        vector = self.process_batch([text], never_stop=False, loop_timeout=180)[0]
        return self.get_items_by_vector(vector, n, include_distances = include_distances, vault = vault if vault else self.vault)


    def get_similar_from_vaults(self, text: str, n: int = 4, vaults = None):
        '''
            Cross-vault similarity search with flexible selection behavior.

            Behavior by type of `vaults`:
            - str: Treat as a single target vault and return top-n from that vault (with distances)
            - list[str]: Search each vault, merge, globally sort by distance, return top-n overall
            - dict[str, int]: Enforce a minimum number from each vault; fill any remaining slots with
              the best overall matches across all vaults.

            Args:
                text: The query text to embed and search with
                n: Total number of items to return overall
                vaults: A string, list of vault names, or a dict mapping vault name -> minimum count

            Returns:
                A list of up to n items, each including a 'distance' field
        '''
        if not vaults:
            raise ValueError("`vaults` must be provided as a str, list[str], or dict[str, int].")

        T(target=self.cloud_manager.build_update, args=(n,)).start()
        vector = self.process_batch([text], never_stop=False, loop_timeout=180)[0]

        # Single vault (string): normal single-vault search with distances
        if isinstance(vaults, str):
            try:
                return self.get_items_by_vector(vector, n, include_distances=True, vault=vaults)
            except Exception:
                return []

        # List of vaults: merge and take global top-n
        if isinstance(vaults, list) and all(isinstance(v, str) for v in vaults):
            combined_results = []
            for v in vaults:
                try:
                    results = self.get_items_by_vector(vector, n, include_distances=True, vault=v)
                    print(f"results: {results}")
                    combined_results.extend(results)
                except Exception:
                    continue
            combined_sorted = sorted(combined_results, key=lambda x: x.get('distance', float('inf')))
            return combined_sorted[:n]

        # Dict of vault -> minimum count
        if isinstance(vaults, dict):
            minima = {k: int(v) for k, v in vaults.items() if isinstance(k, str)}
            total_min = sum(minima.values())

            # 1. If total_min > n, override n to match total_min
            if total_min > n:
                n = total_min

            # 2. If total_min == n, use the minimum for retrieval number, not n
            # 3. If n > total_min, fetch n from each vault for leftovers
            selected = []
            leftovers = []

            # Determine how many to fetch from each vault
            # If n > total_min, fetch n from each vault for leftovers
            # If n <= total_min, fetch min_k from each vault (enough to satisfy minimums)
            fetch_per_vault = {}
            if n > total_min:
                for v, min_k in minima.items():
                    fetch_per_vault[v] = n
            else:
                for v, min_k in minima.items():
                    fetch_per_vault[v] = min_k

            # Retrieve results from each vault
            vault_results = {}
            for v, min_k in minima.items():
                try:
                    results = self.get_items_by_vector(vector, fetch_per_vault[v], include_distances=True, vault=v)
                except Exception:
                    results = []
                vault_results[v] = results

                # Take the top min_k from this vault (or fewer if unavailable)
                take_k = min(min_k, len(results))
                selected.extend(results[:take_k])

                # Keep the rest as leftovers for global fill
                if len(results) > take_k:
                    leftovers.extend(results[take_k:])

            # Fill remaining with best overall from leftovers
            remaining_needed = max(0, n - len(selected))
            if remaining_needed > 0:
                leftovers_sorted = sorted(leftovers, key=lambda x: x.get('distance', float('inf')))
                selected.extend(leftovers_sorted[:remaining_needed])

            # Final global sort for presentation consistency
            selected_sorted = sorted(selected, key=lambda x: x.get('distance', float('inf')))
            return selected_sorted[:n]

        # Unsupported type
        raise ValueError("`vaults` must be a str, list[str], or dict[str, int].")


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


    def add_item_with_vector(self, text: str, vector: list, meta: dict = None, name: str = None):
        """
            If your text length is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        self.check_index()
        start_time = time.time()

        if self._ai.get_tokens(text) > 4000:
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
        if not self.openai_key:
            raise "Cannot get embeddings without an OpenAI API key" 
        
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
            

    def get_chat(self, 
            text: str = None, # The text to send to the LLM
            history: str = '', # The chat history to send to the LLM
            summary: bool = False, # Whether or not the LLM should return a summary of the text
            get_context: bool = False, # Whether or not the LLM should get context from the vault before responding
            n_context: int = 4, # How many items to return from the vault
            return_context: bool = False, # Whether or not the LLM should return the context in the response
            history_search: bool = False, # Whether or not the LLM should search the vector database with chat history as well as the text input
            smart_history_search: bool = False, # Whether or not the LLM should search the chat history for context using a custom prompt
            model: str = None, # The model to use for the LLM
            include_context_meta: bool = False, # Whether or not the LLM should see the context meta data in the context
            custom_prompt: bool = False, # Optionally inject your own custom prompt here
            temperature: int = 0, # The temperature to use for the LLM
            timeout: int = 300, # How long to wait for the LLM to respond
            image_path: str = None, # The path to an image to send to the LLM
            image_url: str = None, # The url of an image to send to the LLM
            vaults: Union[None, str, List[str], Dict] = None, # A vault name, list of vaults, or dict of vaults to search for context in
            conversation_user_id: str = None, # DEPRECATED: No longer used
            ):
        '''
            Chat get response from OpenAI's ChatGPT. 
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
            smart_history_search is False by default skip adding the history of the conversation to the text input for similarity search (useful if history contains subject infomation useful for answering the new text input and the text input doesn't contain that info)
            
            - Example Custom Prompt:
            ```
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
            ```

            Return values:
            - If return_context=False (default): returns the response string
            - If return_context=True: returns a dict with keys 'response' and 'context'

        '''
        self._ensure_prompts_ready()
        start_time = time.time()
        model = self.all_models['default'] if not model else model
        self.load_ai(model=model)
        
        if text: 
            inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = [0]

        response = ''

        if image_path or image_url:
            # Image inference logic
            if self.verbose:
                print("Performing image inference...")

            response = self._ai.platform.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=text,
                model=model,
                temperature=temperature,
                timeout=timeout
            )
            
        else: 
            for segment in inputs:
                attempts = 0
                while True:
                    try:
                        # Make API call
                        if summary and not get_context:
                            token_limit = self._ai.model_token_limit(model)
                            total_tokens = self._ai.get_tokens(segment)
                            response = ""

                            for i in range(math.ceil(total_tokens / token_limit)):
                                # Calculate start and end indices for the current chunk
                                start_index = i * token_limit
                                end_index = min((i + 1) * token_limit, total_tokens)
                                # Extract the current chunk from the segment
                                current_chunk = segment[start_index:end_index]
                                # Process the current chunk and concatenate the response
                                response += ' ' + self._ai.summarize(current_chunk, model=model, custom_prompt=custom_prompt, temperature=temperature)
                                self.rate_limiter.on_success()

                        elif text and get_context and not summary:
                            if smart_history_search:
                                custom_entry = f"Using the current message, with the message history, what subject is the user is focused on. \nCurrent message: {text}. \n\nPrevious messages: {history}."
                                search_input = self._ai.llm(custom_prompt=custom_entry, model=model, temperature=temperature, timeout=timeout)
                            else:
                                search_input = segment + history if history_search else segment
                                
                            if vaults:
                                context = self.get_similar_from_vaults(search_input, n=n_context, vaults=vaults)
                            else:
                                context = self.get_similar(search_input, n=n_context)
                            input_ = str(context) if include_context_meta else ''
                            for text in context:
                                input_ += text['data']

                            response = self._ai.llm_w_context(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                        else: # Custom prompt only
                            if inputs[0] == 0:
                                response = self._ai.llm(model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                            else:
                                response = self._ai.llm(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature, timeout=timeout)
                                
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

        print("get chat time --- %s seconds ---" % (time.time() - start_time)) if self.verbose else 0
            
        return {'response': response, 'context': context, 'search_input': search_input} if return_context else response
        

    def get_chat_stream(self, 
            text: str = None, # The text to send to the LLM
            history: str = '', # The chat history to send to the LLM
            summary: bool = False, # Whether or not the LLM should return a summary of the text
            get_context: bool = False, # Whether or not the LLM should get context from the vault before responding
            n_context: int = 4, # How many items to return from the vault
            return_context: bool = False, # Whether or not the LLM should return the context in the response
            history_search: bool = False, # Whether or not the LLM should search the vector database with chat history as well as the text input
            smart_history_search: bool = False, # Whether or not the LLM should search the chat history for context using a custom prompt
            model: str = None, # The model to use for the LLM
            include_context_meta: bool = False, # Whether or not the LLM should see the context meta data in the context
            metatag: bool = False, # Legacy parameter, do not use
            metatag_prefixes: bool = False, # Legacy parameter, do not use
            metatag_suffixes: bool = False, # Legacy parameter, do not use
            custom_prompt: bool = False, # Optionally inject your own custom prompt here
            temperature: int = 0, # The temperature to use for the LLM
            timeout: int = 300, # How long to wait for the LLM to respond
            image_path: str = None, # The path to an image to send to the LLM
            image_url: str = None, # The url of an image to send to the LLM
            vaults: Union[None, str, List[str], Dict] = None, # A vault name, list of vaults, or dict of vaults to search for context in
            conversation_user_id: str = None, # DEPRECATED: No longer used
            ):
        '''
            Always use this get_chat_stream() wrapped by either print_stream(), or cloud_stream()

            - Example Signle Usage: 
            `response = vault.print_stream(vault.get_chat_stream(text))`

            - Example Context-Response with Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True))`

            - Example Context-Response with Specific Meta Tags for Context Samples Returned:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author']))`
            
            - Example Context-Response with Specific Meta Tags for Context Samples Returned & Specific Meta Prefixes and Suffixes:
            `vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author'], metatag_prefixes=['\n\n Title: ', '\nAuthor: '], metatag_suffixes=['', '\n']))`

            - Example Custom Prompt:
            ```python
            my_prompt = """Answer this question as if you were a financial advisor: "{content}". """
            response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
            ```

            Streaming return payload:
            - If return_context=True, after the token stream finishes, a single JSON string is yielded BEFORE '!END':
              {"response": "<full_response>", "context": [ ... ]}
            - Context items are only streamed inline if metatag parameters are provided.

            How to capture the final context payload in a stream consumer:
            ```python
            # Capture JSON context payload (do not append or emit)
            if isinstance(word, str) and word.startswith('{"response":') and word.endswith('}'):
                try:
                    context_data = json.loads(word)
                    collected_context = context_data.get('context')
                    continue
                except Exception:
                    pass
            else:
                # handle normal token output
                pass
            ```
        '''
        self._ensure_prompts_ready()
        start_time = time.time()
        model = self.all_models['default'] if not model else model
        self.load_ai(model=model)

        if text:
            inputs = [text]
        else:
            if not custom_prompt:
                raise ValueError("No input text provided. Please enter text to proceed.")
            else:
                inputs = []
                
        counter = 0

        if image_path or image_url:
            # Image inference logic
            if self.verbose:
                print("Performing image inference...")

            for i in self._ai.platform.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=text,
                model=model,
                temperature=temperature,
                timeout=timeout
            ):
                yield i
            yield '!END'
            
        else: 
            for segment in inputs:
                start_time = time.time()
                exceptions = 0
                full_response = ''
                while True:
                    try:
                        if summary and not get_context:
                            try:
                                token_limit = self._ai.model_token_limit(model)
                                total_tokens = self._ai.get_tokens(segment)

                                for i in range(math.ceil(total_tokens / token_limit)):
                                    # Calculate start and end indices for the current chunk
                                    start_index = i * token_limit
                                    end_index = min((i + 1) * token_limit, total_tokens)
                                    # Extract the current chunk from the segment
                                    current_chunk = segment[start_index:end_index]
                                    # Process the current chunk and concatenate the response
                                    for word in self._ai.summarize_stream(current_chunk, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                        full_response += word
                                        yield word
                                    yield ' '

                                self.rate_limiter.on_success()
                            except Exception as e:
                                raise e
                            if counter == len(inputs):
                                yield '!END'
                                self.rate_limiter.on_success()
                        
                        elif text and get_context and not summary:
                            if smart_history_search:
                                custom_entry = f"Using the current message, with the message history, what is the user is focused on. \nCurrent message: {text}. \n\nPrevious messages: {history}."
                                search_input = self._ai.llm(custom_prompt=custom_entry, model=model, temperature=temperature, timeout=timeout)
                            else:
                                search_input = segment + history if history_search else segment
                            
                            if vaults:
                                context = self.get_similar_from_vaults(search_input, n=n_context, vaults=vaults)
                            else:
                                context = self.get_similar(search_input, n=n_context)
                            input_ = str(context) if include_context_meta else ''
                            for text in context:
                                input_ += text['data']

                            try:
                                for word in self._ai.llm_w_context_stream(segment, input_, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                    full_response += word
                                    yield word
                                self.rate_limiter.on_success()
                            except Exception as e:
                                raise e

                            if return_context:
                                # Only stream context items if metatag parameters are specified
                                if metatag:
                                    for item in context:
                                        if metatag_prefixes:
                                            if metatag_suffixes:
                                                for i in range(len(metatag)):
                                                    yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}']) + str(metatag_suffixes[i])
                                            else:
                                                for i in range(len(metatag)):
                                                    yield str(metatag_prefixes[i]) + str(item['metadata'][f'{metatag[i]}'])
                                        yield item['data']
                                
                                # Yield the context in the same format as get_chat
                                yield json.dumps({'response': full_response, 'context': context})
                                yield '!END'
                                self.rate_limiter.on_success()
                            else:
                                yield '!END'
                                self.rate_limiter.on_success()

                        else:
                            try:
                                for word in self._ai.llm_stream(segment, history, model=model, custom_prompt=custom_prompt, temperature=temperature):
                                    full_response += word
                                    yield word
                                self.rate_limiter.on_success()
                            except Exception as e:
                                raise e
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
            For printing chat stream. Call 'printing=False' for no pretty printing to be applied
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
            if word == '!END':
                yield f"event: done\n\n"

    

    def run_flow(self, flow_name, message, history: str = '', invoke_method = None, 
                 internal_vars: dict = None, image_url: str = None, **kwargs):
        """
        Returns response from a flow execution.
        
        Args:
            flow_name: Name of the flow to execute
            message: Message to send to the flow
            history: Previous conversation history
            invoke_method: Method to invoke for the flow
            internal_vars: Internal variables to pass to the flow
            
        Returns:
            Full response from the flow execution
        """
        return run_flow(
            user = self.user,
            api_key=self.api,
            flow_name=flow_name,
            message=message,
            history=history,
            conversation_user_id = None,
            invoke_method = invoke_method,
            internal_vars = internal_vars,
            image_url=image_url,
            **kwargs
            )
        
    def stream_flow(self, flow_name, message, history: str = '', invoke_method = None, 
                    internal_vars: dict = None, image_url: str = None, **kwargs):
        """
        Streams response from a flow execution.
        
        Args:
            flow_name: Name of the flow to execute
            message: Message to send to the flow
            history: Previous conversation history
            invoke_method: Method to invoke for the flow
            internal_vars: Internal variables to pass to the flow
            
        Yields:
            Stream events from the flow execution
        """
        # Get the generator from run_flow_stream
        stream_generator = run_flow_stream(
            user = self.user,
            api_key = self.api,
            flow_name = flow_name,
            message = message,
            history = history,
            conversation_user_id = None,
            invoke_method = invoke_method,
            internal_vars = internal_vars,
            image_url=image_url,
            **kwargs
        )
        
        # Yield each event from the generator
        for event in stream_generator:
            yield event
        
    
    def create_storage_dir(self, path: str) -> None:
        """
        Creates a directory at the given path.
        """
        self.storage.create_directory(path)

    def create_storage_item(self, path: str, value: str) -> None:
        """
        Creates a new item (file) at the given path with 'value' as its content.
        """
        self.storage.create_item(path, value)

    def list_storage_labels(self, path: str = None) -> list[dict]:
        """
        Lists all items and directories under 'path'.
        Returns a list of dicts, each dict containing 'name' and 'type' ('directory' or 'item').
        """
        return self.storage.list_labels(path)

    def get_storage_item(self, path: str) -> str:
        """
        Retrieves the text value of an existing item at 'path'.
        """
        return self.storage.get_item(path)

    def update_storage_item(self, path: str, new_value: str) -> None:
        """
        Overwrites the content of an existing item at 'path' with 'new_value'.
        """
        self.storage.update_item(path, new_value)

    def delete_storage_dir(self, path: str) -> None:
        """
        Deletes the specified path (either an item or an entire directory).
        If it's a directory, deletes everything within it recursively.
        """
        self.storage.delete_label(path)


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

