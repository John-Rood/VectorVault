import openai
import tiktoken

class AI:
    def __init__(self) -> None:
        pass

    # This function returns a ChatGPT completion based on a provided input.
    def llm(self, user_input: str = None, history: str = None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False):
        '''
            If you pass in a custom_prompt with content already fully filled in, and no user_input, it will process your custom_prompt only without changing       
        '''
        prompt_template = custom_prompt if custom_prompt else """{content}""" 
        if user_input:
            intokes = self.get_tokens(user_input)
            histokes = self.get_tokens(history) if history else 0
            if intokes + histokes > max_tokens:
                tokes_left = max_tokens - intokes - histokes
                if tokes_left < 0: # way too much input
                    char_to_remove = (tokes_left * -1) * 5 # make positive and remove that amount
                    user_input = user_input[char_to_remove:] # get in front of it, chop at max
                if history:
                    intokes = self.get_tokens(user_input)
                    tokes_left = max_tokens - intokes
                    chars_left = int(tokes_left * 4)
                    history = history[-chars_left:]
                else: # no history. If it was overlimit, then it was taken care of above
                    pass
            if history:
                prompt = prompt_template.format(content=user_input)
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"Chat history: {history}"},
                        {"role": "user", "content": f"{prompt}"}]
                )
                return response['choices'][0]['message']['content']
            else:
                # 'model' is the name of the model to use
                # 'messages' is a list of message objects that mimics a conversation.
                # Each object has a 'role' that can be 'system', 'user', or 'assistant', and a 'content' which is the actual content of the message.
                prompt = prompt_template.format(content=user_input)
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": f"{prompt}"}]
                )
                return response['choices'][0]['message']['content']
        else: # make no changes, and return response to passed in custom_prompt
            if custom_prompt:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": f"{custom_prompt}"}]
                )
                return response['choices'][0]['message']['content']
            else:
                raise 'Error: Need custom_prompt if no user_input'
            
                    
    def llm_w_context(self, user_input, context, history=None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False):
        prompt_template = custom_prompt if custom_prompt else """
        Use the following Context to answer the Question at the end. 
        Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

        Chat History (if any): {history}

        Additional Context: {context}

        Question: {content}

        (Answer the question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
        Answer:""" 

        intokes = self.get_tokens(user_input)
        contokes = self.get_tokens(context)
        history = history if history else ""
        histokes = self.get_tokens(history)
        promptokes = self.get_tokens(prompt_template)

        if (intokes + contokes + histokes + promptokes) > max_tokens * .9:
            tokes_left = max_tokens - intokes
            if len(history) > 1:
                tokes_left = (max_tokens/2) - intokes 
                char_left = int(tokes_left * 4)
                history = history[-char_left:]
                tokes_left_after_hist = max_tokens - self.get_tokens(user_input + history)
                char_left_after_hist = int(tokes_left_after_hist * 4)
                context = context[-char_left_after_hist:]
                double_check = self.get_tokens(user_input+history+context+prompt_template)
                if double_check > max_tokens: 
                    overby = double_check - max_tokens
                    char_to_take_away = overby * 5
                    context_length = len(context)
                    remove_from_context = int(context_length - char_to_take_away)
                    context = context[-remove_from_context:] 
            else:
                char_left = int(tokes_left * 4) 
                context = context[-char_left:]
                double_check = self.get_tokens(user_input + context)
                if double_check > max_tokens:
                    overby = double_check - max_tokens
                    char_to_take_away = overby * 5
                    context_length = len(context)
                    remove_from_context = int(context_length - char_to_take_away)
                    context = context[-remove_from_context:] 

        # Format the prompt
        user_input = history + user_input
        prompt = prompt_template.format(context=context, history=history, content=user_input)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}],
        )
        return response['choices'][0]['message']['content']


    def llm_stream(self, user_input, history=None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False):
        prompt_template = custom_prompt if custom_prompt else """{content}""" 
        intokes = self.get_tokens(user_input)
        histokes = self.get_tokens(history) if history else 0
        if intokes + histokes > max_tokens:
            tokes_left = max_tokens - intokes - histokes
            if tokes_left < 0: # way too much input
                char_to_remove = (tokes_left * -1) * 5 # make positive and remove that amount
                user_input = user_input[char_to_remove:] # get in front of it, chop at max
            if history:
                intokes = self.get_tokens(user_input)
                tokes_left = max_tokens - intokes
                chars_left = int(tokes_left * 4)
                history = history[-chars_left:]
            else: # no history. If it was overlimit, then it was taken care of above
                pass
        if history:
            prompt = prompt_template.format(content=user_input)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Chat history: {history}"},
                    {"role": "user", "content": f"{prompt}"}]
            )
            for message in response:
                choices = message.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    if 'content' in delta:
                        content = delta['content']
                        yield content
        else:
            prompt = prompt_template.format(content=user_input)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}"}],
                stream=True
            )
            for message in response:
                choices = message.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    if 'content' in delta:
                        content = delta['content']
                        yield content
                        
                    
    def llm_w_context_stream(self, user_input, context, history=None, model='gpt-3.5-turbo', max_tokens=4000, custom_prompt=False):
        prompt_template = custom_prompt if custom_prompt else """
        Use the following Context to answer the Question at the end. 
        Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

        Chat History (if any): {history}

        Additional Context: {context}

        Question: {content}

        (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
        Answer:""" 

        intokes = self.get_tokens(user_input)
        contokes = self.get_tokens(context)
        history = history if history else ""
        histokes = self.get_tokens(history)
        promptokes = self.get_tokens(prompt_template)

        if (intokes + contokes + histokes + promptokes) > max_tokens * .9:
            tokes_left = max_tokens - intokes
            if len(history) > 1:
                tokes_left = (max_tokens/2) - intokes 
                char_left = int(tokes_left * 4)
                history = history[-char_left:]
                tokes_left_after_hist = max_tokens - self.get_tokens(user_input + history)
                char_left_after_hist = int(tokes_left_after_hist * 4)
                context = context[-char_left_after_hist:]
                double_check = self.get_tokens(user_input+history+context+prompt_template)
                if double_check > max_tokens: 
                    overby = double_check - max_tokens
                    char_to_take_away = overby * 5
                    context_length = len(context)
                    remove_from_context = int(context_length - char_to_take_away)
                    context = context[-remove_from_context:] 
            else:
                char_left = int(tokes_left * 4) 
                context = context[-char_left:]
                double_check = self.get_tokens(user_input + context)
                if double_check > max_tokens:
                    overby = double_check - max_tokens
                    char_to_take_away = overby * 5
                    context_length = len(context)
                    remove_from_context = int(context_length - char_to_take_away)
                    context = context[-remove_from_context:] 

        prompt = prompt_template.format(context=context, history=history, content=user_input)
        user_input = history + user_input
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}],
            stream=True
        )
        for message in response:
            choices = message.get('choices', [])
            if choices:
                delta = choices[0].get('delta', {})
                if 'content' in delta:
                    content = delta['content']
                    yield content


    def summarize(self, user_input, model='gpt-3.5-turbo', custom_prompt=False):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response['choices'][0]['message']['content']

    def summarize_stream(self, user_input, model='gpt-3.5-turbo', custom_prompt=False):   
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}"}],
            stream = True
        )
        for message in response:
                choices = message.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    if 'content' in delta:
                        content = delta['content']
                        yield content
    
    def smart_summary(self, text, previous_summary, model='gpt-3.5-turbo', custom_prompt=False):   
        prompt_template = custom_prompt if custom_prompt else """Given the previous summary: {previous_summary} 
        Continue from where it leaves off by summarizing the next segment content: {content}"""
        prompt = prompt_template.format(previous_summary=previous_summary, content=text)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response['choices'][0]['message']['content']
        

    def get_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    