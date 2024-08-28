from groq import Groq
import tiktoken
import threading
import queue

stock_sys_msg = "You are an AI assistant that excels at following instructions exactly."

class GroqAPI:
    def __init__(self, personality_message: str = None, main_prompt: str = None, verbose: bool = False, timeout: int = 300, api = None) -> None:
        self.verbose = verbose
        self.timeout = timeout
        self.client = Groq(api_key=api)
        self.default_model = 'llama3-8b-8192'
        self.model_token_limits = {
        'llama3-8b-8192': 8192,
        'llama3-70b-8192': 8192,
        'mixtral-8x7b-32768': 32768,
        'gemma-7b-it': 8192,
    }
        
        self.main_prompt = "Question: {content}" 
        self.main_prompt_with_context = """Use the following Context to answer the Question at the end. 
    Answer as if you were the modern voice of the context, without referencing the context or mentioning 
    the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

    Additional Context: {context}

    Question: {content}
    """ if not main_prompt else main_prompt
        
        self.personality_message = personality_message if personality_message else """Answer directly and be helpful"""
        self.context_prompt = self.main_prompt_with_context + '\n' + f'({self.personality_message})' + '\n' 
        self.prompt = self.main_prompt + '\n\n' + f'({self.personality_message})' + '\n'
        
        
    def within_context_window(self, text : str = None, model='llama3-8b-8192'):
        return self.get_tokens(text) < self.model_token_limits.get(model, 4000)
    
    def model_check(self, token_count, model):
        suitable_models = {model_name: tokens for model_name, tokens in self.model_token_limits.items() if tokens >= token_count}
        
        # If the current model can handle the token count, keep it
        if model in suitable_models:
            return model
        
        else: # Otherwise, switch to the smallest model that can handle the token count
            new_model = min(suitable_models, key=suitable_models.get)
            print('model switch from model:', model, 'to model:', new_model)
            return new_model

    def make_call(self, messages, model, temperature, timeout=None):
        # This function will be run in a separate thread
        timeout = self.timeout if not timeout else timeout
        def call_api(response_queue):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages
                    )
                response_queue.put(response.choices[0].message.content)
            except Exception as e:
                response_queue.put(e)

        response_queue = queue.Queue()
        api_thread = threading.Thread(target=call_api, args=(response_queue,))
        api_thread.start()
        try:
            # Wait for the response with a timeout
            return response_queue.get(timeout=timeout)  # Timeout in seconds
        except queue.Empty:
            # Handle timeout
            print("Request timed out")
            return None


    def llm(self, user_input: str = '', history: str = '', model=None, max_tokens = 8000, custom_prompt = False, temperature = 0, timeout = None, max_retries = 5):        
        '''
            If you pass in a custom_prompt, make sure you format your inputs - this function will not change it
            If you want a custom_prompt but also want to pass `user_input` to take advantage of this function's formatting, then save your custom prompt as default with `save_custom_prompt` in vault.py
        '''
        timeout = self.timeout if not timeout else timeout
        prompt_template = custom_prompt if custom_prompt else self.prompt

        # Use token_model_check to select the suitable model based on token count
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(str(history) + str(user_input) + str(prompt_template)), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if user_input:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), max_tokens=max_tokens)
            user_input, history = new_texts['text'], new_texts['history']

            prompt = prompt_template.format(content=user_input)

        elif custom_prompt:
            prompt = custom_prompt
        else:
            raise ValueError('Error: Need custom_prompt if no user_input')

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        for _ in range(max_retries):
            response = self.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")

            

    def llm_sys(self, content = None, system_message = stock_sys_msg, model = None, max_tokens = 8000, temperature = 0):
        tokens = self.get_tokens(f"{content} {system_message}")
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(system_message + content), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if tokens > max_tokens:
            raise f"Too many tokens: {tokens}"
                
        return self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": content
                }]).choices[0].message.content
    
    
    def llm_instruct(self, content = str, instructions = str, system_message = stock_sys_msg, model = None, max_tokens = 8000, temperature = 0):
        '''
            Give instructions on what to do with the content.
            Usually someone will process content, and the instructions tell how to process it.
        '''
        tokens = self.get_tokens(f"{content} {instructions} {system_message}")
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(system_message + content), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if tokens > max_tokens:
            raise f"Too many tokens: {tokens}"
        
        return self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f'''Follow this instructions you are provided in order to properly process the following content
        Content: {content}
        Instructions: {instructions}'''
                }]).choices[0].message.content


    def llm_w_context(self, user_input = '', context = '', history = '', model = None, max_tokens = 8000, custom_prompt = False, temperature = 0, timeout = None, max_retries = 5):
        timeout = self.timeout if not timeout else timeout
        prompt_template = custom_prompt if custom_prompt else self.context_prompt 
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(str(history) + str(user_input) + str(prompt_template) + str(context)), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if user_input and context:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), str(context), max_tokens=max_tokens)
            user_input, history, context = new_texts['text'], new_texts['history'], new_texts['context']

        # Format the prompt
        prompt = prompt_template.format(context=context, content=user_input)
        print(prompt) if self.verbose == True else 1

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

            
        for _ in range(max_retries):
            response = self.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")


    def llm_stream(self, user_input = '', history = '', model = None, custom_prompt = False, temperature = 0):
        '''
            Stream version of "llm" with dynamic token limit adaptation.
        '''
        prompt_template = custom_prompt if custom_prompt else self.prompt

        # Use token_model_check to select the suitable model based on token count
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(str(history) + str(user_input) + str(prompt_template)), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if user_input:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), max_tokens=max_tokens)
            user_input, history = new_texts['text'], new_texts['history']

            prompt = prompt_template.format(content=user_input)

        elif custom_prompt:
            prompt = custom_prompt
        else:
            raise ValueError('Error: Need custom_prompt if no user_input')


        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 
                        
                    
    def llm_w_context_stream(self, user_input = '', context = '', history = '', model = None, custom_prompt = False, temperature = 0):
        '''
        Function to handle different model sizes automatically and adapt context and history accordingly. 
        '''
        prompt_template = custom_prompt if custom_prompt else self.context_prompt
        model = model if model else self.default_model
        model = self.model_check(self.get_tokens(str(history) + str(user_input) + (prompt_template) + (context)), model)
        max_tokens = self.model_token_limits.get(model, 8000)

        if user_input and context:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), str(context), max_tokens=max_tokens)
            user_input, history, context = new_texts['text'], new_texts['history'], new_texts['context']

        # Format the prompt
        prompt = prompt_template.format(context=context, content=user_input)
        print(prompt) if self.verbose == True else 1

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 


    def summarize(self, user_input, model=None, custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        model = model if model else self.default_model
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response.choices[0].message.content

    def summarize_stream(self, user_input, model=None, custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        model = model if model else self.default_model
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}],
            stream = True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 

    def smart_summary(self, text, previous_summary, model=None, custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Given the previous summary: {previous_summary} 
        Continue from where it leaves off by summarizing the next segment content: {content}"""
        prompt = prompt_template.format(previous_summary=previous_summary, content=text)
        model = model if model else self.default_model
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response.choices[0].message.content
        
    def get_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        try: 
            num_tokens = len(encoding.encode(string))
            return num_tokens
        
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
    
    def truncate_text(self, text, history = '', prompt = '', context = '', max_tokens = 8000): 
        '''
            Enter your text and max tokens, with all other fields optional and get the return
            return is dictionary like: { 'text': text, 'history': history, 'context': context }
        '''

        def cut(text, tokens_to_remove):
            return text[(tokens_to_remove * 4):]
        
        def get_tokes(text, history, prompt, context):
            text_tokens = self.get_tokens(text) if text else 0
            history_tokens = self.get_tokens(history) if history else 0
            prompt_tokens = self.get_tokens(prompt) if prompt else 0
            context_tokens = self.get_tokens(context) if context else 0
            return text_tokens + history_tokens + prompt_tokens + context_tokens
    
        total_tokens = get_tokes(text, history, prompt, context)
        if total_tokens > max_tokens:
            excess_tokens = total_tokens - max_tokens
            if len(history) > max_tokens * 4:
                history = cut(history, excess_tokens)

        total_tokens = get_tokes(text, history, prompt, context)
        if total_tokens > max_tokens:
            excess_tokens = total_tokens - max_tokens
            if len(context) > max_tokens * 4:
                context = cut(context, excess_tokens) 

        total_tokens = get_tokes(text, history, prompt, context)
        if total_tokens > max_tokens:
            excess_tokens = total_tokens - max_tokens
            if len(text) > max_tokens * 4:
                text = cut(text, excess_tokens) 

        return { 'text': text, 'history': history, 'context': context }

