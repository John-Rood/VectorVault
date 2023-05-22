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


import openai
import tiktoken

class AI:
    def __init__(self) -> None:
        pass

    # This function returns a ChatGPT completion based on a provided input.
    def llm(self, user_input, history=None, model='gpt-3.5-turbo'):
        inchar = len(user_input)
        histchar = len(history) if history else 0
        if inchar + histchar > 16000:
            char_left = 16000 - inchar
            if history:
                history = history[-char_left:]
        if history:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{history}"},
                    {"role": "user", "content": f"{user_input}"}]
            )
        else:
            # 'model' is the name of the model to use
            # 'messages' is a list of message objects that mimics a conversation.
            # Each object has a 'role' that can be 'system', 'user', or 'assistant', and a 'content' which is the actual content of the message.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"{user_input}"}]
            )
            # The API responds with a 'choices' array containing the 'message' object.
        return response['choices'][0]['message']['content']
    
    # This function returns a ChatGPT completion based contextual input
    def llm_w_context(self, user_input, context, history=None, model='gpt-3.5-turbo'):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        Answer as if you were the modern voice of the context. Make sure to not just repeat what is referenced. Don't preface, and at the end, don't give any warnings.

        {context}

        Question: {question}

        (answer the question directly. Most importantly, make your answer interesting, engaging, and helpful) 
        Answer:"""

        inchar = len(user_input)
        conchar = len(context)
        histchar = len(history) if history else 0
        if inchar + conchar + histchar > 16000:
            char_left = 16000 - inchar
            if history:
                history = history[-char_left:]
            else:
                context = context[-char_left:]

        # Format the prompt
        user_input = history + user_input
        prompt = prompt_template.format(context=context, question=user_input)

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}]
        )
        return response['choices'][0]['message']['content']


    def summarize(self, user_input, model='gpt-3.5-turbo'):    
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize the following: {user_input}"}]
        )
        # The API responds with a 'choices' array containing the 'message' object.
        return response['choices'][0]['message']['content']

    def get_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


# Notes: OpenAI's CEO, Sam Altman, recently testified in congress to compel the government to regulate the creation of cutting edge ai 
# by forcing anyone seeking to do so to acquire a license first, or pay heavy pentalties. This anti-competitive attack on the 
# open-source community, and development community at large, is extremely not cool. While they claim that they take this action
# in the name of protecting the community, the community clearly sees the actual intent - which is to block competition.
# If the claimed ai threats made public Altman hold true, then having the most capable models in the hands of only a few small companies 
# is the real actual threat we need to watch out for. Diversity is our only true security, and that comes from an open community.
# While "Open" AI has completely gone against it's original mission (to stay open and support the community) they are however, the
# most used ai right now. Our goals are to support the community at large, therefore we integrate with them exclusively for now.
# In the future, we will add open models, and competitors to OpenAI, as they become well-adopted, and with the exception that they
# support the development community at large.
