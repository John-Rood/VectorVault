import openai
import tiktoken
import threading
import queue
import tempfile

stock_sys_msg = "You are an AI assistant that excels at following instructions exactly."

class AI:
    def __init__(self, personality_message: str = None, main_prompt: str = None) -> None:
        self.model_token_limits = {
        'gpt-3.5-turbo': 4000,
        'gpt-3.5-turbo-16k': 16000,
        'gpt-4': 8000,
        'gpt-4-32k': 32000,
        'gpt-4-1106-preview': 128000,
    }
        
        self.main_prompt_with_context = """Use the following Context to answer the Question at the end. 
    Answer as if you were the modern voice of the context, without referencing the context or mentioning 
    the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

    Additional Context: {context}

    Question: {content}
    """ if not main_prompt else main_prompt
        
        self.personality_message = personality_message if personality_message else """Answer directly and be helpful"""
        self.context_prompt = self.main_prompt_with_context + '\n' + f'({self.personality_message})' + '\n\n' + '''Answer:'''
        self.prompt = "Question: {content}" + '\n\n' + f'({self.personality_message})' + '\n\n' + '''Answer:'''
        
    def within_context_window(self, text : str = None, model='gpt-3.5-turbo'):
        return self.get_tokens(text) < self.model_token_limits.get(model, 4000)

    def make_call(self, messages, model, temperature, timeout=45):
        # This function will be run in a separate thread
        def call_api(response_queue):
            try:
                response = openai.chat.completions.create(
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


    # This function returns a ChatGPT completion based on a provided input
    def llm(self, user_input: str = None, history: str = None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False, temperature=0, timeout=45, max_retries=5):        
        '''
            If you pass in a custom_prompt with content already fully filled in, and no user_input, it will process your custom_prompt only without changing anything
            If you pass in a custom_prompt, but also pass in user_input, then it will format the custom_prompt to add the user_input (assumes you did not put the user_input inside the custom_prompt already)
        '''
        prompt_template = custom_prompt if custom_prompt else self.prompt 
        max_tokens = self.model_token_limits.get(model, 4000)
        user_input = '' if user_input is None else user_input
        history = '' if history is None else history
        tokens = self.get_tokens(history + user_input + prompt_template)
        if tokens >= max_tokens:
            if model == 'gpt-3.5-turbo' or model == 'gtp-4':
                model = 'gpt-3.5-turbo-16k'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)
                if tokens > max_tokens:
                    model = 'gpt-4-1106-preview'
                    print('model switch:', model)
                    max_tokens = self.model_token_limits.get(model, 4000)

        if user_input or history:
            intokes = self.get_tokens(user_input) if user_input else 0
            histokes = self.get_tokens(history) if history else 0

            # Calculate the total tokens and determine how many tokens are available
            total_tokens = intokes + histokes

            # Truncate history if the total token count exceeds the max token limit
            if total_tokens > max_tokens:
                excess_tokens = total_tokens - max_tokens
                history = self.truncate_text(history, excess_tokens) if history else None

            # Double check that we are within the limit, if not, it's the user_input
            if self.get_tokens(history + user_input + prompt_template) >= max_tokens:
                user_input = self.truncate_text(user_input, excess_tokens)

            # Construct the prompt with any remaining user input
            history = '' if history is None else history
            prompt = prompt_template.format(content=user_input)

        # If user_input is not provided, and a custom prompt is given, use the custom prompt only
        elif custom_prompt:
            prompt = custom_prompt
        else:
            raise ValueError('Error: Need custom_prompt if no user_input')

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        # Return the call
        for _ in range(max_retries):
            response = self.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")
            

    def llm_sys(self, content = None, system_message = stock_sys_msg, model='gpt-3.5-turbo', max_tokens=4000, temperature=0):
        max_tokens = self.model_token_limits.get(model, 4000)
        tokens = self.get_tokens(f"{content} {system_message}")
        if model == 'gpt-3.5-turbo' or model == 'gtp-4' and tokens > max_tokens:
            model = 'gpt-3.5-turbo-16k'
            print('model switch:', model)
            max_tokens = self.model_token_limits.get(model, 4000)
            if tokens > max_tokens:
                model = 'gpt-4-1106-preview'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)

        if tokens > max_tokens:
            raise f"Too many tokens: {tokens}"
                
        return openai.chat.completions.create(
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
    
    
    def llm_instruct(self, content = str, instructions = str, system_message = stock_sys_msg, model='gpt-3.5-turbo', max_tokens=4000, temperature=0):
        '''
            Give instructions on what to do with the content.
            Usually someone will process content, and the instructions tell how to process it.
        '''
        max_tokens = self.model_token_limits.get(model, 4000)
        tokens = self.get_tokens(f"{content} {instructions} {system_message}")
        if model == 'gpt-3.5-turbo' or model == 'gtp-4' and tokens > max_tokens:
            model = 'gpt-3.5-turbo-16k'
            print('model switch:', model)
            max_tokens = self.model_token_limits.get(model, 4000)
            if tokens > max_tokens:
                model = 'gpt-4-1106-preview'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)

        if tokens > max_tokens:
            raise f"Too many tokens: {tokens}"
        
        return openai.chat.completions.create(
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


    def llm_w_context(self, user_input = None, context = None, history=None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False, temperature=0, timeout=45, max_retries=5):
        prompt_template = custom_prompt if custom_prompt else self.context_prompt
        max_tokens = self.model_token_limits.get(model, 4000)
        user_input = '' if user_input is None else user_input
        history = '' if history is None else history
        tokens = self.get_tokens(history + user_input + context + prompt_template)
        if tokens >= max_tokens:
            if model == 'gpt-3.5-turbo' or model == 'gtp-4':
                model = 'gpt-3.5-turbo-16k'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)
                if tokens > max_tokens:
                    model = 'gpt-4-1106-preview'
                    print('model switch:', model)
                    max_tokens = self.model_token_limits.get(model, 4000)

        if user_input and context:
            # Token calculation for each part.
            intokes = self.get_tokens(user_input)
            contokes = self.get_tokens(context)
            histokes = self.get_tokens(history) if history else 0
            promptokes = self.get_tokens(prompt_template)

            # Calculate the total tokens used and the remaining tokens available.
            total_tokens = intokes + contokes + histokes + promptokes

            # If the total token count exceeds the max token limit, start truncating.
            if total_tokens > max_tokens:
                excess_tokens = total_tokens - max_tokens
                tokens_to_remove = excess_tokens // 2
                    
                history = self.truncate_text(history, tokens_to_remove) if history else None
                context = self.truncate_text(context, tokens_to_remove) if context else None
                
                # Double check that we are within the limit, if not, it's the user_input.
                if self.get_tokens(history + context + user_input + prompt_template) >= max_tokens:
                    user_input = self.truncate_text(user_input, tokens_to_remove * 2)
                    
                # Triple check that we are within the limit, if not, it's the custom prompt.
                if self.get_tokens(history + context + user_input + prompt_template) >= max_tokens:
                    prompt_template = self.truncate_text(prompt_template, tokens_to_remove * 2)

        # Format the prompt
        print(prompt_template)
        prompt = prompt_template.format(context=context, content=user_input)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

            
        for _ in range(max_retries):
            response = self.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")


    def llm_stream(self, user_input=None, history=None, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):
        '''
            Stream version of "llm" with dynamic token limit adaptation.
        '''
        prompt_template = custom_prompt if custom_prompt else self.prompt 

        # Determine the token limit for the selected model.
        max_tokens = self.model_token_limits.get(model, 4000)
        user_input = '' if user_input is None else user_input
        history = '' if history is None else history
        tokens = self.get_tokens(history + user_input + prompt_template)
        if tokens >= max_tokens:
            if model == 'gpt-3.5-turbo' or model == 'gtp-4':
                model = 'gpt-3.5-turbo-16k'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)
                if tokens > max_tokens:
                    model = 'gpt-4-1106-preview'
                    print('model switch:', model)
                    max_tokens = self.model_token_limits.get(model, 4000)

        if user_input:
            # Token calculation for each part.
            intokes = self.get_tokens(user_input)
            histokes = self.get_tokens(history) if history else 0
            promptokes = self.get_tokens(prompt_template)

            # Calculate the total tokens used and the remaining tokens available.
            total_tokens = intokes + histokes + promptokes

            # If the total token count exceeds the max token limit, start truncating.
            if total_tokens > max_tokens:
                excess_tokens = total_tokens - max_tokens
                history = self.truncate_text(history, excess_tokens) if history else 0
                histokes = self.get_tokens(history) if history else 0
                # Double check that we are within the limit.
                if history:
                    assert self.get_tokens(history + user_input + prompt_template) <= max_tokens, "Token limit exceeded."
                else:
                    assert self.get_tokens(user_input + prompt_template) <= max_tokens, "Token limit exceeded."
        
        prompt = prompt_template.format(content=user_input)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 
                        
                    
    def llm_w_context_stream(self, user_input=None, context=None, history=None, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):
        '''
        Function to handle different model sizes automatically and adapt context and history accordingly. 
        '''
        prompt_template = custom_prompt if custom_prompt else self.context_prompt

        # Determine the token limit for the selected model.
        max_tokens = self.model_token_limits.get(model, 4000)
        user_input = '' if user_input is None else user_input
        history = '' if history is None else history

        # Token calculation for each part.
        intokes = self.get_tokens(user_input) if user_input else 0
        contokes = self.get_tokens(context) if context else 0
        histokes = self.get_tokens(history) if history else 0
        promptokes = self.get_tokens(prompt_template)

        # Calculate the total tokens used and the remaining tokens available.
        total_tokens = intokes + contokes + histokes + promptokes

        # If the total token count exceeds the max token limit, start truncating.
        if total_tokens > max_tokens:
            if model == 'gpt-3.5-turbo' or model == 'gtp-4':
                model = 'gpt-3.5-turbo-16k'
                print('model switch:', model)
                max_tokens = self.model_token_limits.get(model, 4000)
                if total_tokens > max_tokens:
                    model = 'gpt-4-1106-preview'
                    print('model switch:', model)
                    max_tokens = self.model_token_limits.get(model, 4000)

            excess_tokens = total_tokens - max_tokens
            tokens_to_remove = excess_tokens // 2
            
            history = self.truncate_text(history, tokens_to_remove) if history else None
            context = self.truncate_text(context, tokens_to_remove) if context else None
            
            # Double check that we are within the limit, if not, it's the user_input.
            if self.get_tokens(history + context + user_input + prompt_template) >= max_tokens:
                user_input = self.truncate_text(user_input, tokens_to_remove * 2)
                
            # Triple check that we are within the limit, if not, it's the custom prompt.
            if self.get_tokens(history + context + user_input + prompt_template) >= max_tokens:
                prompt_template = self.truncate_text(prompt_template, tokens_to_remove * 2)

        # Construct the final prompt
        prompt = prompt_template.format(context=context, content=user_input)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})


        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 


    def summarize(self, user_input, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response.choices[0].message.content

    def summarize_stream(self, user_input, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}],
            stream = True
        )
        for message in response:
            message = message.choices[0].delta.content
            if message:
                yield message 

    def smart_summary(self, text, previous_summary, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):   
        prompt_template = custom_prompt if custom_prompt else """Given the previous summary: {previous_summary} 
        Continue from where it leaves off by summarizing the next segment content: {content}"""
        prompt = prompt_template.format(previous_summary=previous_summary, content=text)
        response = openai.chat.completions.create(
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
    
    def truncate_text(self, text, tokens_to_remove):          
        return text[(tokens_to_remove * 4):]


    def text_to_speech(self, text, model="tts-1", voice="onyx"):
        """
        Creates speech from text using the specified model and voice,
        then saves the output to a temporary file.

        :param text: The text to convert to speech.
        :param model: The speech model to use.
        :param voice: The voice to use.
        :return: The path to the temporary file containing the speech.
        """
        # Create speech response from the client
        response = openai.audio.speech.create(
        model=model,
        voice=voice,
        input=text
        )
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        
        # Stream the response to the temporary file
        response.stream_to_file(temp_file.name)
        
        # Make sure to close the file to flush all writes
        temp_file.close()
        
        # Return the path to the temporary file
        return temp_file

    def transcribe_audio(self, file, model="whisper-1"):
        """
        Transcribes the given audio file using OpenAI's specified model.

        :param file: A file-like object containing the audio to transcribe.
        :param model: The model to use for transcription. Defaults to "whisper-1".
        :return: The transcription result as a string.
        """
        try:
            transcription = openai.audio.transcriptions.create(model=model, file=file)
            return transcription
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None