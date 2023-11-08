import openai
import tiktoken
stock_sys_msg = "You are an AI assistant that excels at following instructions exactly."

class AI:
    def __init__(self) -> None:
        self.model_token_limits = {
        'gpt-3.5-turbo': 4000,
        'gpt-3.5-turbo-16k': 16000,
        'gpt-4-32k': 32000,
        'gpt-4-1106-preview': 128000,
    }
        
    # This function returns a ChatGPT completion based on a provided input.
    def llm(self, user_input: str = None, history: str = None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False, temperature=0):
        '''
            If you pass in a custom_prompt with content already fully filled in, and no user_input, 
            it will process your custom_prompt only without changing       
        '''
        max_tokens = self.model_token_limits.get(model, 4000)
        prompt_template = custom_prompt if custom_prompt else """{content}""" 
        if user_input or history:
            intokes = self.get_tokens(user_input) if user_input else 0
            histokes = self.get_tokens(history) if history else 0

            # Calculate the total tokens and determine how many tokens are available.
            total_tokens = intokes + histokes
            tokens_available = max_tokens - total_tokens

            # Truncate history if the total token count exceeds the max token limit.
            if histokes > tokens_available:
                history = self.truncate_text(history, tokens_available)
                histokes = self.get_tokens(history)
                tokens_available = max_tokens - (intokes + histokes)

            # Construct the prompt with any remaining user input.
            prompt = prompt_template.format(content=user_input or '')

            # Include history in the prompt if it exists.
            if history:
                prompt = f"Chat history: {history}\n\n{prompt}"

        # If user_input is not provided, and a custom prompt is given, use the custom prompt.
        elif custom_prompt:
            prompt = custom_prompt
        else:
            raise ValueError('Error: Need custom_prompt if no user_input')

        # Call the API to get a response.
        return openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
            

    def llm_sys(self, content = None, system_message = stock_sys_msg, model='gpt-3.5-turbo', max_tokens=4000, temperature=0):
        max_tokens = self.model_token_limits.get(model, 4000)
        tokens = self.get_tokens(f"{content} {system_message}")

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
        tokens = self.get_tokens(f"{content} {system_message}")

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


    def llm_w_context(self, user_input = None, context = None, history=None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """
Use the following Context to answer the Question at the end. 
Answer as if you were the modern voice of the context, without referencing the context or mentioning 
the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

Chat History (if any): {history}

Additional Context: {context}

Main Question: {content}

(Answer the Main Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
Answer:""" 
        
        max_tokens = self.model_token_limits.get(model, 4000)
        
        if user_input and context:
            # Token calculation for each part.
            intokes = self.get_tokens(user_input)
            contokes = self.get_tokens(context)
            history = history if history else ""
            histokes = self.get_tokens(history) if history else 0
            promptokes = self.get_tokens(prompt_template)

            # Calculate the total tokens used and the remaining tokens available.
            total_tokens = intokes + contokes + histokes + promptokes
            tokens_available = max_tokens - total_tokens

            # If the total token count exceeds the max token limit, start truncating.
            if total_tokens > max_tokens:
                # Start by truncating history, then context.
                if histokes > tokens_available:
                    history = self.truncate_text(history, tokens_available)
                    histokes = self.get_tokens(history)
                    tokens_available = max_tokens - (intokes + contokes + histokes + promptokes)

                if contokes > tokens_available:
                    context = self.truncate_text(context, tokens_available)
                    contokes = self.get_tokens(context)
                    tokens_available = max_tokens - (intokes + contokes + histokes + promptokes)

                # Double check that we are within the limit.
                assert self.get_tokens(history + context + user_input + prompt_template) <= max_tokens, "Token limit exceeded."

            # Format the prompt
            user_input = history + user_input
            prompt = prompt_template.format(context=context, history=history, content=user_input)

        return openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "user", "content": f"{prompt}"}],
        ).choices[0].message.content


    def llm_stream(self, user_input=None, history=None, model='gpt-3.5-turbo', custom_prompt=False, temperature=0):
        '''
            Stream version of "llm" with dynamic token limit adaptation.
        '''
        prompt_template = custom_prompt if custom_prompt else """{content}""" 

        # Determine the token limit for the selected model.
        max_tokens = self.model_token_limits.get(model, 4000)

        if user_input:
            # Token calculation for each part.
            intokes = self.get_tokens(user_input)
            histokes = self.get_tokens(history) if history else 0
            promptokes = self.get_tokens(prompt_template)

            # Calculate the total tokens used and the remaining tokens available.
            total_tokens = intokes + histokes + promptokes
            tokens_available = max_tokens - total_tokens

            # If the total token count exceeds the max token limit, start truncating.
            if total_tokens > max_tokens:
                # If there's history, prioritize truncating history before input.
                if history and histokes > tokens_available:
                    history = self.truncate_text(history, tokens_available)
                    histokes = self.get_tokens(history)
                    tokens_available = max_tokens - (intokes + histokes + promptokes)

                # Double check that we are within the limit.
                assert self.get_tokens(history + user_input + prompt_template) <= max_tokens, "Token limit exceeded."

            prompt = prompt_template.format(content=user_input)

        if history:
            history_prompt = f"Chat history: {history}"
            prompt = history_prompt + "\n\n" + prompt

        # API call to stream the completion.
        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}],
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
        prompt_template = custom_prompt if custom_prompt else """
Use the following Context to answer the Question at the end. 
Answer as if you were the modern voice of the context, without referencing the context or mentioning 
the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

Chat History (if any): {history}

Additional Context: {context}

Main Question: {content}

(Respond to the Main Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
Answer:"""

        # Determine the token limit for the selected model.
        max_tokens = self.model_token_limits.get(model, 4000)

        # Token calculation for each part.
        intokes = self.get_tokens(user_input) if user_input else 0
        contokes = self.get_tokens(context) if context else 0
        histokes = self.get_tokens(history) if history else 0
        promptokes = self.get_tokens(prompt_template)

        # Calculate the total tokens used and the remaining tokens available.
        total_tokens = intokes + contokes + histokes + promptokes
        tokens_available = max_tokens - total_tokens

        # If the total token count exceeds the max token limit, start truncating.
        if total_tokens > max_tokens:
            # Start by truncating history, then context.
            if histokes > tokens_available:
                history = self.truncate_text(history, tokens_available)
                histokes = self.get_tokens(history)
                tokens_available = max_tokens - (intokes + contokes + histokes + promptokes)

            if contokes > tokens_available:
                context = self.truncate_text(context, tokens_available)
                contokes = self.get_tokens(context)
                tokens_available = max_tokens - (intokes + contokes + histokes + promptokes)

            # Double check that we are within the limit.
            assert self.get_tokens(history + context + user_input + prompt_template) <= max_tokens, "Token limit exceeded."

        # Construct the final prompt.
        prompt = prompt_template.format(context=context, history=history, content=user_input)

        # API call (mocked here, replace with actual API call and handle the response).
        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "user", "content": f"{prompt}"}],
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
    
    def truncate_text(self, text, max_length_in_tokens):
        # A utility method to truncate text to a certain length in tokens.
        tokens = self.get_tokens(text)
        if tokens > max_length_in_tokens:
            # Truncate the text based on the tokens. This is a simplified version.
            # You might need a more sophisticated method to handle tokenization correctly.
            return text[:max_length_in_tokens]
        return text
