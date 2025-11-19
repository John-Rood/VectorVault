import threading
import queue
import tempfile
from abc import ABC, abstractmethod
import openai
import tiktoken
import anthropic
import base64
import httpx
from google import genai
from google.genai import types


# ============================================================================
# MODEL DEFINITIONS - Single Source of Truth
# ============================================================================

OPENAI_MODELS = {
    'o1': 200000,
    'o1-mini': 128000,
    'o3': 200000,
    'o3-pro': 200000,
    'o3-mini': 128000,
    'o4-mini': 200000,
    'gpt-4': 32000,
    'gpt-5': 400000,
    'gpt-5.1': 400000,
    'gpt-5-mini': 400000,
    'gpt-5-nano': 400000,
    'gpt-4o-mini': 128000,
    'gpt-4o': 128000,
    'gpt-4o-audio-preview': 128000,
    'chatgpt-4o-latest': 128000,
    'gpt-5-chat-latest': 400000,
    'gpt-3.5-turbo': 16000,
    'default': 'gpt-5-chat-latest'
}

OPENAI_FRONT_MODELS = {
    'o3': 200000,
    'o3-pro': 200000,
    'o3-mini': 128000,
    'o4-mini': 200000,
    'gpt-5': 400000,
    'gpt-5.1': 400000,
    'gpt-5-mini': 400000,
    'gpt-5-nano': 400000,
    'gpt-4o': 128000,
    'gpt-3.5-turbo': 16000,
    'chatgpt-4o-latest': 128000,
    'gpt-5-chat-latest': 400000,
    'default': 'gpt-5-chat-latest'
}

OPENAI_IMG_CAPABLE = [
    'o1', 'gpt-4o', 'gpt-4o-mini', 'gpt-4o-audio-preview',
    'chatgpt-4o-latest',
    'gpt-5', 'gpt-5.1', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest'
]

OPENAI_NO_STREAM_LIST = ['o1', 'o1-mini']
OPENAI_NO_TEMPERATURE_LIST = ['o1', 'o1-mini', 'o3', 'o3-mini']

GROK_MODELS = {
    "grok-code-fast": 256000,
    "grok-4": 256000,
    "grok-4-fast-reasoning": 2000000,
    "grok-4-fast-non-reasoning": 2000000,
    "grok-3": 131072,
    "grok-3-mini": 131072,
    "grok-2-vision-latest": 32768,
    "default": "grok-4-fast-non-reasoning",
}

GROK_FRONT_MODELS = {
    "grok-code-fast": 256000,
    "grok-4": 256000,
    "grok-4-fast-reasoning": 2000000,
    "grok-4-fast-non-reasoning": 2000000,
    "grok-3": 131072,
    "grok-3-mini": 131072,
    "grok-2-vision-latest": 32768,
    "default": "grok-4-fast-non-reasoning",
}

ANTHROPIC_MODELS = {
    'claude-opus-4-1': 200000,
    'claude-opus-4-0': 200000,
    'claude-sonnet-4-0': 200000,
    'claude-sonnet-4-5': 1000000,
    'claude-3-7-sonnet-latest': 200000,
    'claude-3-5-sonnet-latest': 200000,
    'claude-3-5-haiku-latest': 200000,
    'default': 'claude-sonnet-4-0'
}

ANTHROPIC_FRONT_MODELS = {
    'claude-opus-4-1': 200000,
    'claude-opus-4-0': 200000,
    'claude-sonnet-4-0': 200000,
    'claude-sonnet-4-5': 1000000,
    'claude-3-7-sonnet-latest': 200000,
    'default': 'claude-sonnet-4-5'
}

GEMINI_MODELS = {
    'gemini-2.5-pro': 1000000,
    'gemini-2.5-flash': 1000000,
    'gemini-2.5-flash-lite': 1000000,
    'gemini-2.0-flash': 1000000,
    'default': 'gemini-2.5-flash'
}

GEMINI_FRONT_MODELS = {
    'gemini-2.5-pro': 1000000,
    'gemini-2.5-flash': 1000000,
    'gemini-2.0-flash': 1000000,
    'default': 'gemini-2.5-flash'
}

GEMINI_MULTIMODAL_MODELS = [
    'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash'
]

GEMINI_THINKING_MODELS = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_models(namespaced=False):
    """Get all available models from all platforms"""
    platforms = [
        ('openai', OPENAI_MODELS),
        ('grok', GROK_MODELS),
        ('anthropic', ANTHROPIC_MODELS),
        ('gemini', GEMINI_MODELS),
    ]
    
    all_models = {}
    for platform_name, models in platforms:
        models_copy = models.copy()
        if platform_name != 'openai':
            models_copy.pop('default', None)
        if namespaced:
            models_copy = {f"{platform_name}_{k}": v for k, v in models_copy.items()}
        all_models.update(models_copy)
    return all_models

def get_front_models(namespaced=False):
    """Get front-facing models (subset of all models for UI display)"""
    platforms = [
        ('openai', OPENAI_FRONT_MODELS),
        ('grok', GROK_FRONT_MODELS),
        ('anthropic', ANTHROPIC_FRONT_MODELS),
        ('gemini', GEMINI_FRONT_MODELS),
    ]
    
    front_models = {}
    for platform_name, models in platforms:
        models_copy = models.copy()
        if platform_name != 'openai':
            models_copy.pop('default', None)
        if namespaced:
            models_copy = {f"{platform_name}_{k}": v for k, v in models_copy.items()}
        front_models.update(models_copy)
    return front_models

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

# Platform-agnostic model token limits and default models
class LLMPlatform(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initialize the platform, setting up any required clients, API keys, or
        default model configurations.
        Attributes:
            model_token_limits (dict): A dictionary mapping model names to their maximum token limits.
            default_model (str or None): The model name to use if no model is specified in calls.
        """
        self.model_token_limits = {}
        self.default_model = None

    def list_models(self):
        """
        Best-effort list of available model identifiers for this platform.
        Subclasses may override this to call their native APIs. By default we
        just return the keys from model_token_limits (excluding 'default').
        """
        return [m for m in self.model_token_limits.keys() if m != "default"]

    @abstractmethod
    def make_call(self, messages, model, temperature=None, timeout=None):
        """
        Make a request to the language model with a list of messages and receiv
        a single text-based response
        Args
            messages (list): A list of message dicts. Each dict typically include
                             a "role" (e.g., "system", "user", "assistant") and a "content"
                             If using multimodal features, "content" may be a list of items,
                             e.g., [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}].
            model (str): The name or identifier of the model to call.
            temperature (float, optional): The sampling temperature. Some models
                                           may not support this parameter.
            timeout (int, optional): How long in seconds to wait before timing out.
        Returns:
            str: The text response from the model, or an error string/exception if something fails.
        """
        pass

    @abstractmethod
    def stream_call(self, messages, model, temperature=None, timeout=None):
        """
        Make a streaming request to the language model. Yields tokens or chunks
        of text over time rather than waiting for a single completed response.
        Args:
            messages (list): Same format as make_call(), but may include multimodal content.
            model (str): The model name or identifier.
            temperature (float, optional): The sampling temperature. Some models
                                           may not support this parameter.
            timeout (int, optional): Timeout in seconds.
        Yields:
            str: Pieces of the response as they arrive from the model.
        """
        pass

    @abstractmethod
    def get_tokens(self, string, encoding_name="cl100k_base"):
        """
        Compute the number of tokens in a given string, using a specified tokenizer.
        Args:
            string (str): The text to be tokenized.
            encoding_name (str): The name of the tokenizer/encoding. Defaults to
                                 "cl100k_base", but can differ by model.
        Returns:
            int: The number of tokens in the string.
        """
        pass

    @abstractmethod
    def text_to_speech(self, text, model="tts-1", voice="onyx"):
        """
        Convert text to speech using a TTS model.
        Args:
            text (str): The text to be converted to speech.
            model (str): The TTS model name.
            voice (str): The voice style or ID.
        Returns:
            A file-like object or path to the generated audio (implementation-specific).

        """
        pass

    @abstractmethod
    def transcribe_audio(self, file, model="whisper-1"):
        """
        Transcribe an audio file into text.
        Args:
            file: A file-like object or path to an audio file.
            model (str): The transcription model name.
        Returns:
            str or dict: The transcription result. Exact structure depends on implementation.
        """
        pass

    @abstractmethod
    def model_check(self, token_count, model):
        """
        Given a token_count and a requested model, verify if the request fits
        within that model's token limit. If it doesn't, return a fallback model.
        Args:
            token_count (int): The number of tokens in this request/response.
            model (str): The requested model identifier.
        Returns:
            str: A model name that can handle the token_count. Possibly returns
                 the original model or a fallback/default model.
        """
        pass

    @abstractmethod
    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, timeout=None):
        """
        Handle an image-based request (multimodal). This method should build
        the appropriate messages structure (or accept an already-formed one)
        for sending to a model that supports images.
        Args:
            image_path (str, optional): Path to a local image file to be read/encoded.
            image_url (str, optional): A publicly accessible URL pointing to the image.
            user_text (str, optional): Additional instructions or user prompt.
            model (str, optional): The model to call. Default can be platform.default_model.
            timeout (int, optional): How long to wait before timing out.
        Returns:
            str: The model's textual description/analysis of the provided image.
        """
        pass
    
    @abstractmethod
    def create_embeddings(self, texts, model=None):
        """
        Create embeddings for a list of text inputs.
        Args:
            texts (list[str]): List of text strings to embed.
            model (str, optional): The embeddings model to use. If None, use platform default.
        Returns:
            list[list[float]]: List of embedding vectors.
        """
        pass
    
    @abstractmethod
    def get_embeddings_model_default(self):
        """
        Get the default embeddings model for this platform.
        Returns:
            str: The default embeddings model name.
        """
        pass



# OpenAI Platform Implementation
class OpenAIPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        # Create instance-specific OpenAI client
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        
        # Reference module-level constants
        self.model_token_limits = OPENAI_MODELS
        self.front_model_token_limits = OPENAI_FRONT_MODELS
        self.img_capable = OPENAI_IMG_CAPABLE
        self.no_stream_list = OPENAI_NO_STREAM_LIST
        self.no_temperature_list = OPENAI_NO_TEMPERATURE_LIST
        self.default_model = self.model_token_limits['default']

    def list_models(self):
        """
        Return a list of available OpenAI model IDs.
        Tries the OpenAI models API first, then falls back to known models.
        """
        try:
            models = self.client.models.list()
            data = getattr(models, "data", models)
            ids = []
            for m in data:
                mid = getattr(m, "id", None) or getattr(m, "name", None)
                if mid:
                    ids.append(mid)
            if ids:
                return sorted(set(ids))
        except Exception:
            # Fall back to static model_token_limits if API call fails
            pass
        return super().list_models()

    def make_call(self, messages, model, temperature=None, timeout=None):
        timeout = timeout
        def call_api(response_queue):
            try:
                # Build params
                params = {
                    "model": model,
                    "messages": messages
                }
                # Include temperature only if the model actually supports it
                if temperature is not None and temperature != 0 and model not in self.no_temperature_list:
                    params["temperature"] = temperature
                response = self.client.chat.completions.create(**params)
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
            
    def stream_call(self, messages, model, temperature=None, timeout=None):
        timeout = timeout

        def call_api():
            try:
                # Build params
                params = {
                    "model": model,
                    "messages": messages
                }

                # Conditionally add temperature if the model supports it
                if temperature is not None and temperature != 0 and model not in self.no_temperature_list:
                    params["temperature"] = temperature if temperature else 0

                # If streaming is allowed for this model
                if model not in self.no_stream_list:
                    params["stream"] = True
                    response = self.client.chat.completions.create(**params)
                    for chunk in response:
                        message = chunk.choices[0].delta.content
                        if message:
                            yield message
                else:
                    # Non-streaming fallback
                    response = self.client.chat.completions.create(**params)
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
        response = self.client.audio.speech.create(
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
            transcription = self.client.audio.transcriptions.create(model=model, file=file)
            return transcription
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None

    def model_check(self, token_count, model):
        # Handle 'default' model - return the actual default model name
        if model == 'default':
            return self.model_token_limits['default']
            
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

    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, stream=False, temperature=None, timeout=None):
        """
        Demonstration implementation of an image inference method for OpenAIPlatform.
        For GPT-4o or other multimodal-capable model, you can build the messages 
        payload here and call make_call. If you want to handle images differently,
        adapt as needed.
        """
        model = model if model else self.default_model
        if image_path is None and image_url is None:
            raise ValueError("Must specify either image_path or image_url.")

        if image_path:
            base64_img = encode_image(image_path)
            image_payload = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            }
        else:
            image_payload = {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }

        content = [
            {
                "type": "text",
                "text": user_text if user_text else "Describe this image."
            },
            image_payload
        ]

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return self.make_call(messages, model, temperature=None, timeout=timeout) if not stream else self.stream_call(messages, model, temperature=None, timeout=timeout)
    
    def create_embeddings(self, texts, model=None):
        """
        Create embeddings for a list of text inputs using OpenAI's embeddings API.
        """
        if model is None:
            model = self.get_embeddings_model_default()
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            return [record.embedding for record in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embeddings creation failed: {str(e)}")
    
    def get_embeddings_model_default(self):
        """
        Get the default embeddings model for OpenAI.
        """
        return 'text-embedding-3-small'


###############################################################################
#                            Grok (xAI) Platform                              #
###############################################################################
class GrokPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        # The real client object from the OpenAI library, pointing to xAI's base URL:
        try:
            if api_key is not None:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1",
                )
            else:
                # Set client to None if no API key provided
                self.client = None
        except Exception as e:
            self.client = None
            if api_key:  # Only warn if an API key was explicitly provided
                print(f"Warning: Failed to initialize Grok client: {e}")

        # Reference module-level constants
        self.model_token_limits = GROK_MODELS
        self.front_model_token_limits = GROK_FRONT_MODELS
        self.default_model = self.model_token_limits["default"]

    def list_models(self):
        """
        Return a list of available Grok model IDs.
        Uses the xAI /models endpoint via the OpenAI-compatible client when available.
        """
        # If we don't have a client, just fall back to static list
        if self.client is None:
            return super().list_models()

        try:
            models = self.client.models.list()
            data = getattr(models, "data", models)
            ids = []
            for m in data:
                mid = getattr(m, "id", None) or getattr(m, "name", None)
                if mid:
                    ids.append(mid)
            if ids:
                return sorted(set(ids))
        except Exception:
            pass

        return super().list_models()

    def make_call(self, messages, model, temperature=None, timeout=None):
        """
        Non-streaming call to xAI /v1/chat/completions 
        using the `OpenAI` client set to base_url="https://api.x.ai/v1".
        """
        if self.client is None:
            raise ValueError("Grok client not initialized. Please provide a valid API key.")
            
        def call_api(response_queue):
            try:
                params = {
                    "model": model,
                    "messages": messages,
                }
                # If you need temperature:
                if temperature is not None:
                    params["temperature"] = temperature

                response = self.client.chat.completions.create(**params)
                # xAI returns a structure similar to OpenAI’s
                response_queue.put(response.choices[0].message.content)
            except Exception as e:
                response_queue.put(e)

        response_queue = queue.Queue()
        api_thread = threading.Thread(target=call_api, args=(response_queue,))
        api_thread.start()
        try:
            # Wait for the response up to `timeout` seconds
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            print("Request timed out")
            return None

    def stream_call(self, messages, model, temperature=None, timeout=None):
        """
        Streaming call to xAI /v1/chat/completions 
        with "stream": True.
        """
        if self.client is None:
            raise ValueError("Grok client not initialized. Please provide a valid API key.")
            
        def call_api():
            try:
                params = {
                    "model": model,
                    "messages": messages,
                    "stream": True,  # enable streaming
                }
                
                response = self.client.chat.completions.create(**params)
                for chunk in response:
                    yield chunk.choices[0].delta.content
            except Exception as e:
                yield str(e)

        return call_api()

    def get_tokens(self, string: str, encoding_name: str = None) -> int:
        """
        Naive fallback: approximate token count by len(string)/4.
        """
        return round(len(string) / 4)

    def text_to_speech(self, text, model="default", voice="default"):
        # xAI/Grok does not mention TTS; implement if the API supports it.
        pass

    def transcribe_audio(self, file, model="default"):
        # xAI/Grok does not mention audio transcription; implement if the API supports it.
        pass

    def model_check(self, token_count, model):
        """
        If the requested model can't handle the token_count,
        switch to the smallest model that can, else default.
        """
        if model not in self.model_token_limits:
            return model
        suitable_models = {
            m_name: limit
            for m_name, limit in self.model_token_limits.items()
            if isinstance(limit, int) and limit >= token_count
        }
        if model in suitable_models:
            return model
        else:
            if suitable_models:
                new_model = min(suitable_models, key=suitable_models.get)
                print('model switch from model:', model, 'to model:', new_model)
                return new_model
            else:
                return self.model_token_limits['default']

    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, stream=False, temperature=None, timeout=None):
        """
        For vision-capable models like "grok-2-vision-latest", we can send:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": <URL>, "detail": "high"}
                    },
                    {
                        "type": "text",
                        "text": "Describe what's in this image."
                    }
                ]
            }
        ]
        """
        model = model if model else "grok-2-vision-latest"
        if not image_path and not image_url:
            raise ValueError("Must specify either image_path or image_url.")

        # Build the "content" array
        if image_path:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            image_payload = {
                "type": "image_url",
                "image_url": {
                    # xAI typically expects a real URL, but can handle base64 data:
                    "url": f"data:image/png;base64,{data}",
                    "detail": "high"
                }
            }
        else:
            image_payload = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"
                }
            }

        content = [
            image_payload,
            {
                "type": "text",
                "text": user_text if user_text else "Describe this image."
            }
        ]

        messages = [{"role": "user", "content": content}]
        
        # Use provided temperature or default to 0
        temp = temperature if temperature is not None else 0
        
        if stream:
            return self.stream_call(messages, model=model, temperature=temp, timeout=timeout)
        else:
            return self.make_call(messages, model=model, temperature=temp, timeout=timeout)
    
    def create_embeddings(self, texts, model=None):
        """
        Create embeddings for text inputs. Grok uses OpenAI-compatible API.
        """
        if model is None:
            model = self.get_embeddings_model_default()
        
        # For now, Grok doesn't have embeddings, so raise an error
        raise NotImplementedError("Grok platform does not yet support embeddings")
    
    def get_embeddings_model_default(self):
        """
        Get the default embeddings model for Grok.
        """
        # Grok doesn't have embeddings models yet
        return None


# Anthropic (Claude) Platform Implementation
class AnthropicPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        # Initialize client - will use environment variable ANTHROPIC_API_KEY if api_key is None
        try:
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                # Try to initialize from environment variable
                self.client = anthropic.Anthropic()
        except Exception as e:
            # If initialization fails, set client to None and handle in methods
            self.client = None
            if api_key:  # Only warn if an API key was explicitly provided
                print(f"Warning: Failed to initialize Anthropic client: {e}")

        # Reference module-level constants
        self.model_token_limits = ANTHROPIC_MODELS
        self.front_model_token_limits = ANTHROPIC_FRONT_MODELS
        self.default_model = self.model_token_limits['default']

    def list_models(self):
        """
        Return a list of available Anthropic model IDs.
        Uses the Anthropic models API when a client is available.
        """
        if self.client is None:
            return super().list_models()

        try:
            models = self.client.models.list()
            data = getattr(models, "data", models)
            ids = []
            for m in data:
                mid = getattr(m, "id", None) or getattr(m, "name", None)
                if mid:
                    ids.append(mid)
            if ids:
                return sorted(set(ids))
        except Exception:
            pass

        return super().list_models()

    def make_call(self, messages, model, temperature, timeout=None):
        if self.client is None:
            raise ValueError("Anthropic client not initialized. Please provide a valid API key or set ANTHROPIC_API_KEY environment variable.")
            
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
        if self.client is None:
            raise ValueError("Anthropic client not initialized. Please provide a valid API key or set ANTHROPIC_API_KEY environment variable.")
            
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

    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, stream=False, temperature=None, timeout=None):
        model = model if model else self.default_model

        if image_path is None and image_url is None:
            raise ValueError("Must specify either image_path or image_url.")

        if image_path:
            image_data = encode_image(image_path)
            media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            image_payload = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            }
        else:
            # Fetch and encode image from URL
            response = httpx.get(image_url)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch image from URL: {image_url}")
            media_type = response.headers.get("Content-Type", "image/jpeg")
            image_data = base64.b64encode(response.content).decode("utf-8")
            image_payload = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            }

        content = [
            image_payload,
            {
                "type": "text",
                "text": user_text if user_text else "Describe this image."
            }
        ]

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        if stream:
            return self.stream_call(messages, model, temperature, timeout=timeout)
        else:
            return self.make_call(messages, model, temperature, timeout=timeout)
    
    def create_embeddings(self, texts, model=None):
        """
        Create embeddings for text inputs using Anthropic's API.
        Note: Anthropic doesn't currently offer embeddings models.
        """
        raise NotImplementedError("Anthropic platform does not currently support embeddings")
    
    def get_embeddings_model_default(self):
        """
        Get the default embeddings model for Anthropic.
        """
        # Anthropic doesn't have embeddings models
        return None



# Google Gemini Platform Implementation
class GeminiPlatform(LLMPlatform):
    def __init__(self, api_key=None):
        # Initialize client - will use environment variable GOOGLE_API_KEY if api_key is None
        self.client = None
        self._client_initialized = False
        
        try:
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                # Try to initialize from environment variable
                self.client = genai.Client()
            self._client_initialized = True
        except Exception as e:
            # If initialization fails, set client to None and handle in methods
            self.client = None
            self._client_initialized = False
            if api_key:  # Only warn if an API key was explicitly provided
                print(f"Warning: Failed to initialize Gemini client: {e}")

        # Reference module-level constants
        self.model_token_limits = GEMINI_MODELS
        self.front_model_token_limits = GEMINI_FRONT_MODELS
        self.default_model = self.model_token_limits['default']
        self.multimodal_models = GEMINI_MULTIMODAL_MODELS
        self.thinking_models = GEMINI_THINKING_MODELS
    
    def list_models(self):
        """
        Return a list of available Gemini model IDs.
        Uses the Google GenAI models API when a client is available.
        """
        if self.client is None:
            return super().list_models()

        try:
            models = self.client.models.list()
            # google-genai models typically expose a 'name' attribute
            ids = []
            for m in models:
                mid = getattr(m, "name", None) or getattr(m, "id", None)
                if mid:
                    ids.append(mid)
            if ids:
                return sorted(set(ids))
        except Exception:
            pass

        return super().list_models()
    
    def close(self):
        """
        Explicitly close the Gemini client to clean up resources.
        This can be called manually to ensure proper cleanup.
        """
        if self._client_initialized and self.client is not None:
            try:
                # Check if the client has the close method and _api_client attribute
                if hasattr(self.client, 'close') and hasattr(self.client, '_api_client'):
                    self.client.close()
                    self._client_initialized = False
            except (AttributeError, Exception):
                # Silently handle any cleanup errors
                pass
    
    def __del__(self):
        """
        Cleanup method to safely close the Gemini client.
        This prevents AttributeError when the Google GenAI Client.__del__ 
        tries to access _api_client that may not exist.
        """
        self.close()

    def make_call(self, messages, model, temperature=None, timeout=None):
        def call_api(response_queue):
            try:
                # Check if client is properly initialized
                if self.client is None:
                    raise ValueError("Gemini client not initialized. Please provide a valid API key or set GOOGLE_API_KEY environment variable.")
                
                # Convert messages to Gemini format
                contents = self._convert_messages_to_contents(messages)
                
                # Build config
                config_params = {}
                if temperature is not None:
                    config_params['temperature'] = temperature
                
                # Create config object if we have parameters
                config = types.GenerateContentConfig(**config_params) if config_params else None
                
                # Make the API call
                params = {
                    "model": model,
                    "contents": contents
                }
                if config:
                    params["config"] = config
                    
                response = self.client.models.generate_content(**params)
                response_queue.put(response.text)
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

    def stream_call(self, messages, model, temperature=None, timeout=None):
        def call_api():
            try:
                # Check if client is properly initialized
                if self.client is None:
                    raise ValueError("Gemini client not initialized. Please provide a valid API key or set GOOGLE_API_KEY environment variable.")
                
                # Convert messages to Gemini format
                contents = self._convert_messages_to_contents(messages)
                
                # Build config
                config_params = {}
                if temperature is not None:
                    config_params['temperature'] = temperature
                
                # Create config object if we have parameters
                config = types.GenerateContentConfig(**config_params) if config_params else None
                
                # Make the streaming API call
                params = {
                    "model": model,
                    "contents": contents
                }
                if config:
                    params["config"] = config
                    
                response = self.client.models.generate_content_stream(**params)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield str(e)

        return call_api()

    def _convert_messages_to_contents(self, messages):
        """Convert OpenAI-style messages to Gemini contents format"""
        contents = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Skip system messages for now (can be handled with system_instruction in config)
            if role == 'system':
                continue
                
            # For simple text content
            if isinstance(content, str):
                contents.append(content)
            else:
                # Handle multimodal content (list of parts)
                contents.append(content)
                
        return contents

    def get_tokens(self, string: str, encoding_name: str = None) -> int:
        """
        Get token count for Gemini models using the official count_tokens API.
        Falls back to estimation if client is not available.
        """
        if self.client is not None:
            try:
                response = self.client.models.count_tokens(
                    model=self.default_model,
                    contents=string
                )
                return response.total_tokens
            except Exception:
                # Fall back to estimation if API call fails
                pass
        
        # Fallback estimation: 1 token per 4 characters
        return round(len(string) / 4)

    def text_to_speech(self, text, model="default", voice="default"):
        # Gemini doesn't currently support TTS in the provided API
        raise NotImplementedError("Gemini platform does not currently support text-to-speech.")

    def transcribe_audio(self, file, model="default"):
        # Gemini supports audio input but transcription would be through generate_content
        # with audio files - not a dedicated transcription endpoint
        raise NotImplementedError("Use image_inference method with audio files for Gemini audio processing.")

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

    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, stream=False, temperature=None, timeout=None):
        """
        Handle image inference using Gemini's multimodal capabilities.
        """
        # Check if client is properly initialized
        if self.client is None:
            raise ValueError("Gemini client not initialized. Please provide a valid API key or set GOOGLE_API_KEY environment variable.")
        
        model = model if model else self.default_model
        
        if model not in self.multimodal_models:
            raise ValueError(f"Model {model} does not support image inputs.")
            
        if image_path is None and image_url is None:
            raise ValueError("Must specify either image_path or image_url.")

        # Prepare the content list
        contents = []
        
        # Add text prompt
        if user_text:
            contents.append(user_text)
        else:
            contents.append("Describe this image.")
            
        # Add image
        if image_path:
            # Read local image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Determine MIME type
            mime_type = "image/jpeg"
            if image_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_path.lower().endswith('.gif'):
                mime_type = "image/gif"
            elif image_path.lower().endswith('.webp'):
                mime_type = "image/webp"
                
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            contents.append(image_part)
            
        elif image_url:
            # Fetch image from URL
            import requests
            response = requests.get(image_url)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch image from URL: {image_url}")
                
            # Determine MIME type from Content-Type header or URL
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            if 'image/' not in content_type:
                content_type = 'image/jpeg'  # Default fallback
                
            image_part = types.Part.from_bytes(
                data=response.content,
                mime_type=content_type
            )
            contents.append(image_part)

        # Build config
        config_params = {}
        if temperature is not None:
            config_params['temperature'] = temperature
            
        config = types.GenerateContentConfig(**config_params) if config_params else None

        # Make the API call
        try:
            params = {
                "model": model,
                "contents": contents
            }
            if config:
                params["config"] = config
                
            if stream:
                def stream_generator():
                    response = self.client.models.generate_content_stream(**params)
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return stream_generator()
            else:
                response = self.client.models.generate_content(**params)
                return response.text
                
        except Exception as e:
            raise Exception(f"Gemini image inference failed: {str(e)}")
    
    def create_embeddings(self, texts, model=None):
        """
        Create embeddings for text inputs using Gemini's text embedding models.
        """
        if model is None:
            model = self.get_embeddings_model_default()
        
        try:
            # Gemini uses a different API for embeddings
            results = []
            for text in texts:
                response = self.client.models.embed_content(
                    model=model,
                    content=text
                )
                results.append(response['embedding'])
            return results
        except Exception as e:
            raise Exception(f"Gemini embeddings creation failed: {str(e)}")
    
    def get_embeddings_model_default(self):
        """
        Get the default embeddings model for Gemini.
        """
        return 'models/text-embedding-004'




# LLM Client that uses the platform-agnostic interface
class LLMClient:
    def __init__(self, platform: LLMPlatform, personality_message: str = None, main_prompt: str = None, main_prompt_with_context: str = None, verbose: bool = False, timeout: int = 300, fine_tuned_context_window=128000):
        self.platform = platform
        self.verbose = verbose
        self.default_model = self.platform.default_model
        self.timeout = timeout
        self.fine_tuned_context_window = fine_tuned_context_window
        self.main_prompt = main_prompt if main_prompt else "{content}"
        self.main_prompt_with_context = main_prompt_with_context if main_prompt_with_context else """Use the following Context to answer the Question at the end.
    Answer as if you were the modern voice of the context, without referencing the context or mentioning
    the fact that any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

    Additional Context: {context}

    Question: {content}
    """    
        self.personality_message = personality_message if personality_message else ""
        self.context_prompt = self.main_prompt_with_context + '\n' + f'{self.personality_message}' + '\n'
        self.prompt = self.main_prompt + '\n\n' + f'({self.personality_message})' + '\n'

    @staticmethod
    def _format_prompt(template, **values):
        class _SafeDict(dict):
            def __missing__(self, key):
                return ''
        try:
            return template.format_map(_SafeDict(**values))
        except Exception:
            return template

    def set_prompts(self):
        self.context_prompt = self.main_prompt_with_context + '\n' + f'{self.personality_message}' + '\n'
        self.prompt = self.main_prompt + '\n\n' + f'{self.personality_message}' + '\n'

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

    def text_llm(self, user_input: str = '', history: str = '', model=None, custom_prompt=False, temperature=0, timeout=None, max_retries=5):
        timeout = self.timeout if not timeout else timeout
        prompt_template = custom_prompt if custom_prompt else self.prompt

        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if user_input:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), max_tokens=max_tokens)
            user_input, history = new_texts['text'], new_texts['history']
            prompt = self._format_prompt(prompt_template, content=user_input)
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

    def smart_llm(self, user_input: str = '', history: str = '', model=None, custom_prompt=False, temperature=0, timeout=None, max_retries=5, image_path=None, image_url=None):
        """
        Smart LLM method that automatically switches between text-only LLM and image inference
        based on whether image parameters are provided.
        """
        # If image parameters are provided, use image inference
        if image_path or image_url:
            return self.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=user_input,
                model=model,
                stream=False,
                temperature=temperature,
                timeout=timeout
            )
        else:
            # Otherwise use regular text LLM
            return self.text_llm(
                user_input=user_input,
                history=history,
                model=model,
                custom_prompt=custom_prompt,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries
            )
        
    def llm(self, user_input: str = '', history: str = '', model=None, custom_prompt=False, temperature=0, timeout=None, max_retries=5, image_path=None, image_url=None):
        """
        Smart LLM method that automatically switches between text-only LLM and image inference
        based on whether image parameters are provided.
        """
        # If image parameters are provided, use image inference
        if image_path or image_url:
            return self.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=user_input,
                model=model,
                stream=False,
                temperature=temperature,
                timeout=timeout
            )
        else:
            # Otherwise use regular text LLM
            return self.text_llm(
                user_input=user_input,
                history=history,
                model=model,
                custom_prompt=custom_prompt,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries
            )

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

        prompt = self._format_prompt(prompt_template, context=context, content=user_input)
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
            prompt = self._format_prompt(prompt_template, content=user_input)
        elif custom_prompt:
            prompt = custom_prompt
        else:
            raise ValueError('Error: Need custom_prompt if no user_input')

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def smart_llm_stream(self, user_input='', history='', model=None, custom_prompt=False, temperature=0, image_path=None, image_url=None):
        """
        Smart LLM streaming method that automatically switches between text-only streaming and image inference streaming
        based on whether image parameters are provided.
        """
        # If image parameters are provided, use image inference streaming
        if image_path or image_url:
            yield from self.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=user_input,
                model=model,
                stream=True,
                temperature=temperature
            )
        else:
            # Otherwise use regular text LLM streaming
            yield from self.llm_stream(
                user_input=user_input,
                history=history,
                model=model,
                custom_prompt=custom_prompt,
                temperature=temperature
            )

    def llm_w_context_stream(self, user_input='', context='', history='', model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else self.context_prompt
        model = model if model else self.default_model
        token_count = self.platform.get_tokens(str(history) + str(user_input) + str(prompt_template) + str(context))
        model = self.platform.model_check(token_count, model)
        max_tokens = self.platform.model_token_limits.get(model) if model in self.platform.model_token_limits.keys() else self.fine_tuned_context_window

        if user_input and context:
            new_texts = self.truncate_text(user_input, str(history), str(prompt_template), str(context), max_tokens=max_tokens)
            user_input, history, context = new_texts['text'], new_texts['history'], new_texts['context']

        prompt = self._format_prompt(prompt_template, context=context, content=user_input)
        if self.verbose:
            print(prompt)

        messages = [{"role": "user", "content": history}] if history else []
        messages.append({"role": "user", "content": prompt})

        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def summarize(self, user_input, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = self._format_prompt(prompt_template, content=user_input)
        model = model if model else self.default_model
        messages = [{"role": "user", "content": f"{prompt}"}]
        response = self.platform.make_call(messages, model, temperature)
        return response

    def summarize_stream(self, user_input, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Summarize the following: {content}"""
        prompt = self._format_prompt(prompt_template, content=user_input)
        model = model if model else self.default_model
        messages = [{"role": "user", "content": f"{prompt}"}]
        for message in self.platform.stream_call(messages, model, temperature):
            if message:
                yield message

    def smart_summary(self, text, previous_summary, model=None, custom_prompt=False, temperature=0):
        prompt_template = custom_prompt if custom_prompt else """Given the previous summary: {previous_summary}
Continue from where it leaves off by summarizing the next segment content: {content}"""
        prompt = self._format_prompt(prompt_template, previous_summary=previous_summary, content=text)
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
    
    def image_inference(self, image_path=None, image_url=None, user_text=None, model=None, stream=False, temperature=None, timeout=None):
        """
        Perform image inference using the underlying platform's image_inference method.

        Args:
            image_path (str, optional): Path to a local image file.
            image_url (str, optional): URL of an image.
            user_text (str, optional): Prompt or instructions for the model.
            model (str, optional): Model to use. Defaults to platform's default model.
            stream (bool, optional): Whether to stream the response.
            temperature (float, optional): Sampling temperature.
            timeout (int, optional): Timeout in seconds.

        Returns:
            str or generator: Model's response as text or a generator if streaming.
        """
        model = model if model else self.default_model
        token_count = self.platform.get_tokens(user_text if user_text else "")
        model = self.platform.model_check(token_count, model)

        if stream:
            return self.platform.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=user_text,
                model=model,
                stream=True,
                temperature=temperature,
                timeout=timeout
            )
        else:
            return self.platform.image_inference(
                image_path=image_path,
                image_url=image_url,
                user_text=user_text,
                model=model,
                stream=False,
                temperature=temperature,
                timeout=timeout
            )
        
    def model_token_limit(self, model=None):
        """
        Get the token limit for the specified model or the default model.
        
        Args:
            model (str, optional): The model identifier to get the token limit for.
                                If not provided, uses the default model.
        
        Returns:
            int: The maximum token limit for the specified model or a default
                context window size if the model is not found in the platform's
                token limits dictionary.
        """
        model = model if model else self.default_model
        
        # Check if the model exists in the platform's model_token_limits dictionary
        if model in self.platform.model_token_limits:
            # Return the token limit for the model
            return self.platform.model_token_limits[model]
        else:
            # If model not found, return the default fine-tuned context window size
            return self.fine_tuned_context_window


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
