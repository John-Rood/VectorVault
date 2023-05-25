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
# from Vector Vault.

import numpy as np
import tempfile
import os
import time
import json
import re
import openai
from concurrent.futures import ThreadPoolExecutor
from .cloudmanager import CloudManager
from .ai import AI
from .itemize import itemize, name, name_vecs, get_item, build_return, get_vectors


class Vault:
    def __init__(self, user: str, api_key: str, vault: str = None, dims: int = 1536, verbose: bool = False):
        self.vault = vault.strip() if vault else 'home'
        self.vectors = get_vectors(dims)
        self.dims = dims
        try:
            self.cloud_manager = CloudManager(user, api_key, self.vault)
        except:
            print('API KEY FAILED! Using vault without cloud access. `get_chat()` will still work')
            # user can still use the get_chat() function without an api key
            pass
        self.x = 0
        self.x_checked = False
        self.verbose = verbose
        self.items = []
        self.last_time = None
        self.last_chat_time = None
        self.first_run = True
        self.needed_sleep_time = None
        self.ai = AI()


    def check_index(self):
        start_time = time.time()
        if self.cloud_manager.vault_exists(name_vecs(self.vault)):
            self.load_vectors()
            num_existing_items = self.vectors.get_n_items()
            new_index = get_vectors(self.dims)
            for i in range(num_existing_items):
                vector = self.vectors.get_item_vector(i)
                new_index.add_item(i, vector)
            self.x = i + 1
            self.vectors = new_index
        else:
            pass
        
        self.x_checked = True
        if self.verbose == True:
            print("initialize index --- %s seconds ---" % (time.time() - start_time))
        

    def load_vectors(self):
        start_time = time.time()
        # Download the .ann file from google cloud storage to a temporary file
        temp_file_path = self.cloud_manager.download_to_temp_file(name_vecs(self.vault))

        # Load the AnnoyIndex object from the temporary file, then delete the temp file
        self.vectors.load(temp_file_path)
        os.remove(temp_file_path)
        if self.verbose:
            print("get load vectors --- %s seconds ---" % (time.time() - start_time))

    
    def get_vaults(self, vault: str = None):
        vault = self.vault if vault is None else vault
        return self.cloud_manager.get_vaults(vault)


    def get_total_vectors(self):
        return self.vectors.get_n_items()
    

    def save(self, trees=16):
        start_time = time.time()
        self.vectors.build(trees)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.vectors.save(temp_file.name)
            self.cloud_manager.upload_temp_file(temp_file.name, name_vecs(self.vault))

        total_saved_items = 0
        for item in self.items:
            item_text, item_id, item_meta = get_item(item)
            self.cloud_manager.upload(item_id, item_text, item_meta)
            total_saved_items += 1

        self.items.clear()
        self.x_checked = False

        if self.verbose:
            print("save vectors time --- %s seconds ---" % (time.time() - start_time))


    def delete(self):
        if self.verbose == True:
            print('Deleting started. Note: this can take a while for large datasets')
        # Clear the local vector data
        self.vectors = get_vectors(self.dims)
        self.items.clear()
        self.x = 0
        self.cloud_manager.delete()
        print('Vault deleted')
    

    def split_text(self, text, min_threshold=1000, max_threshold=16000):
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

        last_sentence = text[sentence_start:]
        if last_sentence:
            current_segment.append(last_sentence)

        if current_segment:
            segments.append(" ".join(current_segment))

        return segments


    def get_items_by_vector(self, vector, n: int = 4):
        self.load_vectors()
        start_time = time.time()

        results = []
        vecs = self.vectors.get_nns_by_vector(vector, n)
        for vec in vecs:
            # Retrieve the item
            item_data = self.cloud_manager.download_text_from_cloud(name(self.vault, vec, item=True))
            # Retrieve the metadata
            meta_data = self.cloud_manager.download_text_from_cloud(name(self.vault, vec, meta=True))
            meta = json.loads(meta_data)
            build_return(results, item_data, meta)

        if self.verbose == True:
            print(f"get {n} items back --- %s seconds ---" % (time.time() - start_time))

        return results
    

    def get_similar(self, text, n: int = 4):
        vector = self.process_batch([text], never_stop=False, loop_timeout=180)[0]
        return self.get_items_by_vector(vector, n)


    def add_item(self, text: str, meta: dict = None, name: str = None):
        """
            If your text length lenght is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        if self.x_checked == False:
            self.check_index()
        else: 
            pass

        self.items.append(itemize(self.vault, self.x, meta, text, name))
        self.x += 1


    def add(self, text: str, meta: dict = None, name: str = None):
        """
            If your text length lenght is greater than 4000 tokens, Vault.split_text(your_text)  
            will automatically be added
        """

        if self.ai.get_tokens(text) > 4000:
            if self.verbose == True:
                print('Text length too long. Using the built-in "split_text()" function to get a list of text segments') 
            texts = self.split_text(text) # returns list of text segments
        else:
            texts = [text]

        for text in texts:
            self.add_item(text, meta, name)

    
    def add_item_with_vector(self, text: str, vector: list, meta: dict = None, name: str = None):
        """
            If your text length lenght is greater than 15000 characters, you should use Vault.split_text(your_text) to 
            get a list of text segments that are the right size
        """
        if self.x_checked == False:
            self.check_index()
        else: 
            pass
        start_time = time.time()

        if self.ai.get_tokens(text) > 4000:
            raise 'Text length too long. Use the "split_text() function to get a list of text segments'

        # Add vector to vectorspace
        self.vectors.add_item(self.x, vector)
        self.items.append(itemize(self.vault, self.x, meta, text, name))

        self.x += 1

        if self.verbose == True:
            print("add item time --- %s seconds ---" % (time.time() - start_time))


    def process_batch(self, batch_text_chunks, never_stop, loop_timeout):
        loop_start_time = time.time()
        while True:
            try:
                res = openai.Embedding.create(input=batch_text_chunks, engine="text-embedding-ada-002")
                break
            except Exception as e:
                print(f"API Error: {e}. Sleeping 15 seconds")
                time.sleep(15)
                if not never_stop or (time.time() - loop_start_time) > loop_timeout:
                    try:
                        res = openai.Embedding.create(input=batch_text_chunks, engine="text-embedding-ada-002")
                        break
                    except Exception as e:
                        raise TimeoutError("Loop timed out")
        return [record['embedding'] for record in res['data']]


    def get_vectors(self, batch_size: int = 32, never_stop: bool = False, loop_timeout: int = 180):
        start_time = time.time()
        if not self.last_time:
            self.last_time = start_time - 1
        
        if not self.needed_sleep_time:
            self.needed_sleep_time = 0
        
        time.sleep(self.needed_sleep_time)
        
        texts = []
        text_len = 0
        for item in self.items:
            text = item['text']
            text_len += self.ai.get_tokens(text)
            texts.append(text)
        num_batches = int(np.ceil(len(texts) / batch_size))

        # Prepare the text chunks for all batches
        batches_text_chunks = [
            texts[i * batch_size:min((i + 1) * batch_size, len(texts))]
            for i in range(num_batches)
        ]
        
        # max 350,000 tokens per minute - max requests per minute = 3500
        if self.first_run == False:
            trip_time = float(start_time - self.last_time)
            req_min = 60 / trip_time # 1 min (60) / time between requests (trip_time)
            projected_tokens_per_min = req_min * text_len
            rate_ratio = projected_tokens_per_min / 350000
            if self.verbose == True:
                print(f'Projected Tokens per min:{projected_tokens_per_min} | Rate Limit Ratio: {rate_ratio} | Text Length: {text_len}')
            # 1 min divided by the cap per min and the total we are sending now and factor in the last trip time
            self.needed_sleep_time = 60 / (350000 / text_len) - trip_time 
            if self.needed_sleep_time < 0:
                self.needed_sleep_time = 0

            if self.verbose == True:
                print(f"Time calc'd to sleep: {self.needed_sleep_time}")
            if req_min > 3500:
                time.sleep(1)

        # Process the batches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            process_batch_with_params = lambda batch_text_chunks: self.process_batch(batch_text_chunks, never_stop, loop_timeout)
            batch_embeddings_list = list(executor.map(process_batch_with_params, batches_text_chunks))


        # Add the embeddings to the AnnoyIndex using the index from the metadata
        current_item_index = 0
        for batch_embeddings in batch_embeddings_list:
            for embedding in batch_embeddings:
                item_index = self.items[current_item_index]["meta"]["item_id"]
                self.vectors.add_item(item_index, embedding)
                current_item_index += 1

        self.last_time = start_time
        self.first_run = False
        if self.verbose == True:
            print("get vectors time --- %s seconds ---" % (time.time() - start_time))


    def get_chat(self, text: str, history: str = None, summary: bool = False, get_context = False, n_context = 4, return_context = False, expansion = False, history_search = False, model='gpt-3.5-turbo', include_context_meta=False):
        '''
            Chat get response from OpenAI's ChatGPT. 
            Rate limiting, auto retries, and chat histroy slicing built-in so you can chat with ease. 
            Enter your text, add optional chat history, and optionally choose a summary response (default: summmary = False)

            Example Signle Usage: 
            `response = vault.get_chat(text)`

            Example Chat: 
            `response = vault.get_chat(text, chat_history)`
            
            Example Summary: 
            `summary = vault.get_chat(text, summary=True)`

            Example Context-Based Response:
            `response = vault.get_chat(text, get_context = True)`

            Example Context-Based Response w/ Chat History:
            `response = vault.get_chat(text, chat_history, get_context = True)`

            Example Context-Response with Context Samples Returned:
            `vault_response = vault.get_chat(text, get_context = True, return_context = True)`
            
            Response is a string, unless return_context == True, then response will be a dictionary 

            Example to print dictionary results:
            # print response:
            `print(vault_response['response'])` 

            # print context:
            for item in vault_response['context']['results']:
                print("\n\n", f"item {item['metadata']['item_index']}")
                print(item['data'])

            Default `expansion = False` can be set to True to create additional context from user input for vector retrieval. Allowing for greater search accuracy if
            user input is too short or lacks the specificity needed for a quality retrieval search. ('expansion' is not context-aware). Default is good.

            history_search is False by default skip adding the history of the conversation to the question to retrieval search 
        '''

        start_time = time.time()
        if not self.last_chat_time:
            self.last_chat_time = start_time - 20
        
        if not self.needed_sleep_time:
            self.needed_sleep_time = 0
        
        if not history:
            history = ''
        
        time.sleep(self.needed_sleep_time)

        if self.ai.get_tokens(text) > 4000:
            if summary:
                inputs = self.split_text(text, 14500)
            else:
                inputs = self.split_text(text)
        else:
            inputs = [text]
        response = ''
        for segment in inputs:
            start_time = time.time()
            seg_len = self.ai.get_tokens(segment)
            # max 90,000 tokens per minute | max requests per minute = 3500
            trip_time = float(start_time - self.last_chat_time)
            req_min = 60 / trip_time # 1 min (60) / time between requests (trip_time)
            projected_tokens_per_min = req_min * seg_len
            rate_ratio = projected_tokens_per_min / 90000
            if self.verbose == True:
                print(f'Projected Tokens per min:{projected_tokens_per_min} | Rate Limit Ratio: {rate_ratio} | Text Length: {seg_len}')
            # 1 min divided by the cap per min and the total we are sending now and factor in the last trip time
            if seg_len == 0:
                raise('No input. Add text input to continue')
            self.needed_sleep_time = 60 / (90000 / seg_len) - trip_time 
            if self.needed_sleep_time < 0:
                self.needed_sleep_time = 0
            if self.verbose == True:
                print(f"Time calc'd to sleep: {self.needed_sleep_time}")

            if expansion:
                iq = f"be direct and short. Question: {segment} \n The intent of this question is to: "
                intent_expansion = self.ai.llm(iq)
                kq = f"be general, direct, and short. Don't give an answer, only topics this question falls under to this question: {segment}"
                knowledge_expansion = self.ai.llm(kq)
                segment = f'question_intent: {intent_expansion} | {knowledge_expansion}\n\
                Question: {segment}'

            while True:
                try:
                    if summary and not get_context:
                        response += self.ai.summarize(segment, model=model)
                    elif get_context and not summary:
                        user_input = segment + history if history_search else segment
                        if self.ai.get_tokens(user_input) > 4000:
                            user_input = user_input[-16000:]
                        if self.ai.get_tokens(user_input) > 4000:
                            user_input = user_input[-15000:]
                        if include_context_meta:
                            context = self.get_similar(user_input, n=n_context)
                            input = str(context)
                        else:
                            context = self.get_similar(user_input, n=n_context)
                            input = ''
                            for text in context:
                                input += text['data']
                        response = self.ai.llm_w_context(segment, input, history, model=model)
                    else:
                        response = self.ai.llm(segment, history, model=model)
                    break
                except Exception as e:
                    print(f"API Error: {e}. Sleeping 15 seconds")
                    time.sleep(15)
                    
            self.last_chat_time = start_time

        if self.verbose == True:
            print("get chat time --- %s seconds ---" % (time.time() - start_time))

        if return_context == False:
            return response
        else:
            return {'response': response, 'context': context}
    
