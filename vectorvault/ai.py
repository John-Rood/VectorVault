import threading
import queue
import tempfile
from abc import ABC, abstractmethod
import openai
import tiktoken
import anthropic
import groq


def get_all_models(namespaced=False):
    platforms = [
        ('openai', OpenAIPlatform()),
        ('groq', GroqPlatform()),
        ('anthropic', AnthropicPlatform()),
    ]
    all_models = {}
    for platform_name, platform in platforms:
        models = platform.model_token_limits.copy()
        if not isinstance(platform, OpenAIPlatform):
            models.pop('default', None)
        if namespaced:
            models = {f"{platform_name}_{k}": v for k, v in models.items()}
        all_models.update(models)
    return all_models

def get_front_models(namespaced=False):
    platforms = [
        ('openai', OpenAIPlatform()),
        ('groq', GroqPlatform()),
        ('anthropic', AnthropicPlatform()),
    ]
    front_models = {}
    for platform_name, platform in platforms:
        models = platform.front_model_token_limits.copy()
        if not isinstance(platform, OpenAIPlatform):
            models.pop('default', None)
        if namespaced:
            models = {f"{platform_name}_{k}": v for k, v in models.items()}
        front_models.update(models)
    return front_models


# Platform-agnostic model token limits and default models
class LLMPlatform(ABC):
    @abstractmethod
    def __init__(self):
        self.model_token_limits = {}
        self.default_model = None

    @abstractmethod
    def make_call(self, messages, model, temperature, timeout=None):
        pass

    @abstractmethod
    def stream_call(self, messages, model, temperature, timeout=None):
        pass

    @abstractmethod
    def get_tokens(self, string, encoding_name="cl100k_base"):
        pass

    @abstractmethod
    def text_to_speech(self, text, model="tts-1", voice="onyx"):
        pass

    @abstractmethod
    def transcribe_audio(self, file, model="whisper-1"):
        pass

    @abstractmethod
    def model_check(self, token_count, model):
        pass

# OpenAI Platform Implementation
class OpenAIPlatform(LLMPlatform):
    def __init__(self):
        self.model_token_limits = {
            'o1': 200000,
            'o1-2024-12-17': 200000,
            'o1-mini': 128000,
            'o1-preview': 128000,
            'o1-preview-2024-09-12': 128000,
            'o1-mini-2024-09-12': 128000,
            'gpt-4-turbo': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4o': 128000,
            'gpt-4o-2024-08-06': 128000,
            'gpt-4-turbo-preview': 128000,
            'gpt-4-1106-preview': 128000,
            'gpt-4-0125-preview': 128000,
            'gpt-4-0314': 8000,
            'gpt-4-0613': 8000,
            'gpt-4': 8000,
            'gpt-3.5-turbo': 16000,
            'gpt-3.5-turbo-0125': 16000,
            'default': 'gpt-4o'
        }
        self.front_model_token_limits = {
            'o1': 128000,
            'o1-mini': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4o': 128000,
            'gpt-4': 8000,
            'default': 'gpt-4o'
        }
        self.different_inference_list = ['o1', 'o1-2024-12-17', 'o1-preview', 'o1-mini', 'o1-preview-2024-09-12', 'o1-mini-2024-09-12']
        self.default_model = self.model_token_limits['default']

    def make_call(self, messages, model, temperature, timeout=None):
        # This function will be run in a separate thread
        timeout = timeout
        def call_api(response_queue):
            if model not in self.different_inference_list:
                try:
                    response = openai.chat.completions.create(
                        model=model,
                        temperature=temperature if temperature else 0,
                        messages=messages
                    )
                    response_queue.put(response.choices[0].message.content)
                except Exception as e:
                    response_queue.put(e)
            else:
                try:
                    response = openai.chat.completions.create(
                        model=model,
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

    def stream_call(self, messages, model, temperature, timeout=None):
        timeout = timeout

        def call_api():
            if model not in self.different_inference_list:
                try:
                    response = openai.chat.completions.create(
                        model=model,
                        temperature=temperature if temperature else 0,
                        messages=messages,
                        stream=True
                    )
                    for chunk in response:
                        message = chunk.choices[0].delta.content
                        if message:
                            yield message
                except Exception as e:
                    yield str(e)
            else:
                try:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                    yield response.choices[0].message.content
                except Exception as e:
                    yield str(e)

        return call_api()

    def get_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        try:
            num_tokens = len(encoding.encode(string))
            return num_tokens
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

    def text_to_speech(self, text, model="tts-1", voice="onyx"):
        response = openai.Audio.create(
            model=model,
            voice=voice,
            input=text
        )
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        response.stream_to_file(temp_file.name)
        temp_file.close()
        return temp_file

    def transcribe_audio(self, file, model="whisper-1"):
        try:
            transcription = openai.Audio.transcribe(model=model, file=file)
            return transcription
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None

    def model_check(self, token_count, model):
        if model not in self.model_token_limits.keys():
            return model
        else:
            # Filter out the 'default' key and create suitable_models dictionary
            suitable_models = {
                model_name: tokens 
                for model_name, tokens in self.model_token_limits.items() 
                if isinstance(tokens, int) and tokens >= token_count
            }
            
            if model in suitable_models:
                return model
            else:
                if suitable_models:  # Check if we found any suitable models
                    new_model = min(suitable_models, key=suitable_models.get)
                    print('model switch from model:', model, 'to model:', new_model)
                    return new_model
                else:
                    # If no suitable models found, return the default model
                    return self.model_token_limits['default']


# Groq Platform Implementation
class GroqPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        if api_key: 
            self.client = groq.Client(api_key=api_key)

        self.model_token_limits = {
            'llama3-8b-8192': 8192,
            'llama3-70b-8192': 8192,
            'mixtral-8x7b-32768': 32768,
            'gemma-7b-it': 8192,
            'default': 'llama3-8b-8192'
        }
        self.front_model_token_limits = {
            'llama3-8b-8192': 8192,
            'llama3-70b-8192': 8192,
            'mixtral-8x7b-32768': 32768,
            'gemma-7b-it': 8192,
            'default': 'llama3-8b-8192'
        }
        self.default_model = self.model_token_limits['default']

    def make_call(self, messages, model, temperature, timeout=None):
        timeout = timeout
        def call_api(response_queue):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature if temperature else 0,
                    messages=messages
                )
                response_queue.put(response.choices[0].message.content)
            except Exception as e:
                response_queue.put(e)

        response_queue = queue.Queue()
        api_thread = threading.Thread(target=call_api, args=(response_queue,))
        api_thread.start()
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            print("Request timed out")
            return None

    def stream_call(self, messages, model, temperature, timeout=None):
        def call_api():
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature if temperature else 0,
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    message = chunk.choices[0].delta.content
                    if message:
                        yield message
            except Exception as e:
                yield str(e)

        return call_api()

    def get_tokens(self, string: str, encoding_name: str = None) -> int:
        # Implement token counting for Groq
        # Assuming similar to OpenAI
        # Since no encoding specified, we can estimate tokens
        # For simplicity, assume 1 token per 4 characters
        num_tokens = len(string) / 4
        return int(num_tokens)

    def text_to_speech(self, text, model="default", voice="default"):
        # Implement text-to-speech if available
        pass

    def transcribe_audio(self, file, model="default"):
        # Implement audio transcription if available
        pass

    def model_check(self, token_count, model):
        if model not in self.model_token_limits.keys():
            return model
        else:
            # Filter out the 'default' key and create suitable_models dictionary
            suitable_models = {
                model_name: tokens 
                for model_name, tokens in self.model_token_limits.items() 
                if isinstance(tokens, int) and tokens >= token_count
            }
            
            if model in suitable_models:
                return model
            else:
                if suitable_models:  # Check if we found any suitable models
                    new_model = min(suitable_models, key=suitable_models.get)
                    print('model switch from model:', model, 'to model:', new_model)
                    return new_model
                else:
                    # If no suitable models found, return the default model
                    return self.model_token_limits['default']

# Anthropic (Claude) Platform Implementation
class AnthropicPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model_token_limits = {
            'claude-3-5-sonnet-20241022': 200000,
            'claude-3-5-sonnet-latest': 200000,
            'claude-3-5-haiku-20241022': 200000,
            'claude-3-5-haiku-latest': 200000,
            'default': 'claude-3-5-sonnet-latest'
        }
        self.front_model_token_limits = {
            'claude-3-5-sonnet-latest': 200000,
            'default': 'claude-3-5-sonnet-20241022'
        }
        self.default_model = self.model_token_limits['default']

    def make_call(self, messages, model, temperature, timeout=None):
        def call_api(response_queue):
            try:
                response = self.client.messages.create(
                    model=model,
                    messages=messages,  # Already in the correct format
                    temperature=temperature if temperature else 0,
                    max_tokens=8192
                )
                response_queue.put(response.content[0].text)
            except Exception as e:
                response_queue.put(e)

        response_queue = queue.Queue()
        api_thread = threading.Thread(target=call_api, args=(response_queue,))
        api_thread.start()
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            print("Request timed out")
            return None

    def stream_call(self, messages, model, temperature, timeout=None):
        def call_api():
            try:
                response = self.client.messages.create(
                    model=model,
                    messages=messages,  # Already in the correct format
                    temperature=temperature if temperature else 0,
                    max_tokens=8192,
                    stream=True
                )


                for chunk in response:
                    if chunk.type == 'content_block_delta':
                        yield chunk.delta.text
                    elif chunk.type == 'message_stop':
                        break

            except Exception as e:
                yield str(e)

        return call_api()

    def get_tokens(self, string: str, encoding_name: str = None) -> int:
        # Anthropic uses a similar tokenization method
        # For estimation, assume 1 token per 4 characters
        num_tokens = len(string) / 4
        return int(num_tokens)

    def text_to_speech(self, text, model="default", voice="default"):
        # Implement text-to-speech if available
        pass

    def transcribe_audio(self, file, model="default"):
        # Implement audio transcription if available
        pass

    def model_check(self, token_count, model):
        if model not in self.model_token_limits.keys():
            return model
        else:
            # Filter out the 'default' key and create suitable_models dictionary
            suitable_models = {
                model_name: tokens 
                for model_name, tokens in self.model_token_limits.items() 
                if isinstance(tokens, int) and tokens >= token_count
            }
            
            if model in suitable_models:
                return model
            else:
                if suitable_models:  # Check if we found any suitable models
                    new_model = min(suitable_models, key=suitable_models.get)
                    print('model switch from model:', model, 'to model:', new_model)
                    return new_model
                else:
                    # If no suitable models found, return the default model
                    return self.model_token_limits['default']
        

# LLM Client that uses the platform-agnostic interface
class LLMClient:
    def __init__(self, platform: LLMPlatform, personality_message: str = None, main_prompt: str = None, verbose: bool = False, timeout: int = 300, fine_tuned_context_window=128000):
        self.platform = platform
        self.verbose = verbose
        self.default_model = self.platform.default_model
        self.timeout = timeout
        self.fine_tuned_context_window = fine_tuned_context_window
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

    def set_prompts(self):
        self.context_prompt = self.main_prompt_with_context + '\n' + f'({self.personality_message})' + '\n'
        self.prompt = self.main_prompt + '\n\n' + f'({self.personality_message})' + '\n'

    def within_context_window(self, text: str = None, model=None):
        if model not in self.platform.model_token_limits.keys():
            return self.platform.get_tokens(text) < self.platform.model_token_limits.get(model) if model else self.fine_tuned_context_window
        else:
            return self.platform.get_tokens(text) < self.fine_tuned_context_window

    def truncate_text(self, text, history='', prompt='', context='', max_tokens=128000):
        def cut(text, tokens_to_remove):
            return text[(tokens_to_remove * 4):]

        def get_tokes(text, history, prompt, context):
            text_tokens = self.platform.get_tokens(text) if text else 0
            history_tokens = self.platform.get_tokens(history) if history else 0
            prompt_tokens = self.platform.get_tokens(prompt) if prompt else 0
            context_tokens = self.platform.get_tokens(context) if context else 0
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

        return {'text': text, 'history': history, 'context': context}

    def llm(self, user_input: str = '', history: str = '', model=None, custom_prompt=False, temperature=0, timeout=None, max_retries=5):
        timeout = self.timeout if not timeout else timeout
        prompt_template = custom_prompt if custom_prompt else self.prompt

        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

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
            response = self.platform.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")

    def llm_sys(self, content=None, system_message="You are an AI assistant that excels at following instructions exactly.", model=None, temperature=0):
        tokens = self.platform.get_tokens(f"{content} {system_message}")
        model = model if model else self.default_model
        model = self.platform.model_check(tokens, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if tokens > max_tokens:
            raise Exception(f"Too many tokens: {tokens}")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content}
        ]
        response = self.platform.make_call(messages, model, temperature)
        return response

    def llm_instruct(self, content: str, instructions: str, system_message="You are an AI assistant that excels at following instructions exactly.", model=None, temperature=0):
        tokens = self.platform.get_tokens(f"{content} {instructions} {system_message}")
        model = model if model else self.default_model
        model = self.platform.model_check(tokens, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if tokens > max_tokens:
            raise Exception(f"Too many tokens: {tokens}")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""Follow these instructions to properly process the following content:
Content: {content}
Instructions: {instructions}"""}
        ]
        response = self.platform.make_call(messages, model, temperature)
        return response

    def llm_w_context(self, user_input='', context='', history='', model=None, custom_prompt=False, temperature=0, timeout=None, max_retries=5):
        timeout = self.timeout if not timeout else timeout
        prompt_template = custom_prompt if custom_prompt else self.context_prompt
        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template) + str(context))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if user_input and context:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), str(context), max_tokens=max_tokens)
            user_input, history, context = new_texts['text'], new_texts['history'], new_texts['context']

        prompt = prompt_template.format(context=context, content=user_input)
        if self.verbose:
            print('Full input prompt:', prompt)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        for _ in range(max_retries):
            response = self.platform.make_call(messages, model, temperature, timeout)
            if response is not None:
                return response
            print("Retrying...")

        raise Exception("Failed to receive response within the timeout period.")

    def llm_stream(self, user_input='', history='', model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else self.prompt

        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

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

        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def llm_w_context_stream(self, user_input='', context='', history='', model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else self.context_prompt
        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template) + str(context))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if user_input and context:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), str(context), max_tokens=max_tokens)
            user_input, history, context = new_texts['text'], new_texts['history'], new_texts['context']

        prompt = prompt_template.format(context=context, content=user_input)
        if self.verbose:
            print(prompt)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def summarize(self, user_input, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        model = model if model else self.default_model
        messages = [{"role": "user", "content": f"{prompt}"}]
        response = self.platform.make_call(messages, model, temperature)
        return response

    def summarize_stream(self, user_input, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = prompt_template.format(content=user_input)
        model = model if model else self.default_model
        messages = [{"role": "user", "content": f"{prompt}"}]
        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def smart_summary(self, text, previous_summary, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Given the previous summary: {previous_summary}
Continue from where it leaves off by summarizing the next segment content: {content}"""
        prompt = prompt_template.format(previous_summary=previous_summary, content=text)
        model = model if model else self.default_model
        messages = [{"role": "user", "content": f"{prompt}"}]
        response = self.platform.make_call(messages, model, temperature)
        return response

    def text_to_speech(self, text, model="tts-1", voice="onyx"):
        return self.platform.text_to_speech(text, model, voice)

    def transcribe_audio(self, file, model="whisper-1"):
        return self.platform.transcribe_audio(file, model)
    
    def get_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Get token count for a string using the platform's tokenizer"""
        return self.platform.get_tokens(string, encoding_name)


class ModelsProperty:
    @property
    def front_models(self):
        return get_front_models()
    
    @property
    def all_models(self):
        return get_all_models()

models = ModelsProperty()

front_models = models.front_models
all_models = models.all_models