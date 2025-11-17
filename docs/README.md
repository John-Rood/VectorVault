![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)
# Vector Vault Python SDK

Vector Vault provides a Python SDK for building persistent AI agents that combine vector search, multi-model orchestration, and stateful workflows. This guide covers the complete API surface—from basic RAG operations to advanced agent deployment.

## Architecture Overview

Vector Vault consists of three integrated layers:

1. **Vault Layer**: Managed vector storage with automatic embedding, similarity search, and metadata filtering
2. **Chat Layer**: Multi-provider LLM orchestration with streaming, context injection, and conversation management  
3. **Flow Layer**: Persistent agent runtime for complex, long-running workflows built in Vector Flow

All three layers share the same authentication and can be mixed freely—use vaults for RAG, chat for conversations, and flows for autonomous agents.

## Table of Contents

### Getting Started
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Advantages](#advantages)
- [Use Cases](#use-cases)

### Python API Documentation
- [Installation](#install)
- [Initialization & API Keys](#initialization--api-keys)
  - [Basic Setup (OpenAI Only)](#basic-setup-openai-only)
  - [Multi-Platform Setup (All Providers)](#multi-platform-setup-all-providers)
  - [API Key Requirements](#api-key-requirements)
  - [Platform-Specific Models](#platform-specific-models)
  - [Getting API Keys](#getting-api-keys)
  - [Advanced Initialization Options](#advanced-initialization-options)

### Core Operations
- [Upload Data](#upload)
- [Vault Management](#vault-management)
  - [Creating Vaults](#creating-vaults)
- [Search Your Data](#search-your-data)
- [Talk to Your Data](#talk-to-your-data)
  - [The get_chat() Function](#the-get_chat-function)
  - [Basic Usage](#basic-usage)
  - [Change the Personality](#change-the-personality)

### Advanced Features
- [Summarize Anything](#summarize-anything)
- [Streaming](#streaming)
- [Metadata Made Easy](#metadata-made-easy)
  - [Add Any Fields](#add-any-fields)
- [Vaults](#vaults)
- [Build an AI Customer Service Chatbot](#build-an-ai-cusomter-service-chatbot)
- [Advanced Features](#advanced-features)
  - [Multi-Platform Model Switching](#multi-platform-model-switching)
  - [Image Processing](#image-processing)
  - [Vector Vault Flows](#vector-vault-flows) 
  - [Storage Management](#storage-management)
  - [Data Export/Import](#data-exportimport)

### Support
- [Contact](#contact)

---

## Why Vector Vault SDK?

The Python SDK provides production-ready infrastructure for AI applications:

### Technical Advantages
- **Sub-second RAG pipeline**: Vector search + context injection + streaming in 200-400ms typical
- **Automatic retries & rate limiting**: Built-in exponential backoff for all API calls
- **Lazy initialization**: Connect in microseconds, resources load on first use
- **Type-safe responses**: Consistent data structures across sync/async operations
- **Zero-config embeddings**: Automatic batching and parallel processing for text-embedding-ada-002

### Operational Benefits
- **No infrastructure**: Skip Redis, Pinecone, Weaviate setup and maintenance
- **Unified billing**: One invoice for vectors, agents, and hosting
- **Built-in observability**: Request IDs and timing in responses for debugging
- **Tenant isolation**: Each vault is a completely separate namespace
- **Global edge deployment**: Automatic routing to nearest datacenter

### Scale Characteristics
Based on production workloads across thousands of projects:
- **Embedding throughput**: 10K-50K chunks/minute depending on size
- **Vector operations**: ~5,000 searches/sec and ~1,000 upserts/sec per project
- **Concurrent clients**: Tested with 10,000+ simultaneous connections


<br>
<br>
<br>
<br>

# Full Python API:

Install:
<br>
`pip install vector-vault` 

# Initialization & API Keys

Vector Vault supports multiple AI platforms. You can provide API keys for any combination of providers:

## Basic Setup (OpenAI Only)
```python
from vectorvault import Vault

vault = Vault(user='your_email', 
              api_key='your_vectorvault_api_key',
              openai_key='your_openai_api_key',
              vault='any_vault_name')
```

## Multi-Platform Setup (All Providers)
```python
from vectorvault import Vault

vault = Vault(user='your_email', 
              api_key='your_vectorvault_api_key',
              openai_key='your_openai_api_key',      # For GPT models, embeddings
              grok_key='your_grok_api_key',          # For xAI's Grok models  
              anthropic_key='your_anthropic_api_key', # For Claude models
              gemini_key='your_gemini_api_key',      # For Google Gemini models
              vault='any_vault_name',
              verbose=False,
              model='gpt-4o')  # Set your preferred default model
```

## API Key Requirements

### Required
- **`api_key`**: Your Vector Vault API key (always required)
- **`openai_key`**: Required for embeddings and OpenAI models

### Optional (for multi-platform support)
- **`grok_key`**: Access xAI's Grok models for creative tasks
- **`anthropic_key`**: Use Claude models for reasoning and analysis
- **`gemini_key`**: Access Google Gemini models for multimodal AI tasks

## Platform-Specific Models

```python
# OpenAI models (requires openai_key)
vault.get_chat("Hello", model="gpt-4o")
vault.get_chat("Hello", model="gpt-4o-mini") 
vault.get_chat("Hello", model="o1")

# Grok models (requires grok_key)  
vault.get_chat("Hello", model="grok-4")
vault.get_chat("Hello", model="grok-3")

# Claude models (requires anthropic_key)
vault.get_chat("Hello", model="claude-sonnet-4-0")
vault.get_chat("Hello", model="claude-sonnet-4-5")

# Gemini models (requires gemini_key)
vault.get_chat("Hello", model="gemini-2.5-pro")
vault.get_chat("Hello", model="gemini-2.5-flash")
```

## Getting API Keys

- **Vector Vault**: Sign up at [vectorvault.io](https://vectorvault.io) for 30-day free trial
- **OpenAI**: Get your key at [platform.openai.com](https://platform.openai.com)
- **xAI (Grok)**: Get access at [console.x.ai](https://console.x.ai)
- **Anthropic**: Apply for API access at [console.anthropic.com](https://console.anthropic.com)
- **Google Gemini**: Get your API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

## Advanced Initialization Options

```python
vault = Vault(
    user='your_email',
    api_key='your_vectorvault_api_key',
    openai_key='your_openai_key',
    grok_key='your_grok_key',        # Optional: For Grok models
    anthropic_key='your_anthropic_key',  # Optional: For Claude models
    gemini_key='your_gemini_key',    # Optional: For Gemini models
    vault='vault_name',
    
    # Performance & Behavior
    verbose=True,                    # Enable detailed logging
    embeddings_model='text-embedding-3-large',  # Use larger embedding model
    
    # Conversation Features  
    conversation_user_id='user123',  # Enable conversation history
    model='claude-sonnet-4-5',       # Set default model
    
    # Custom Prompting (overrides saved prompts/personality)
    main_prompt="Answer: {content}",
    main_prompt_with_context="Context: {context}\nQuestion: {content}",
    personality_message="Be concise and helpful"
)
```

**Lazy Initialization**: Vector Vault uses lazy initialization for improved performance. Cloud connections, AI platforms, and vectors are only initialized when first accessed, making vault creation nearly instant. 

`vault.add("text string", meta=None, name=None, split=False, split_size=1000, max_threshold=16000)` : Loads data to be added to the Vault, with automatic text splitting for long texts. Can include metadata and custom name. `split=True` forces text splitting.
<br>
`vault.get_vectors()` : Retrieves vectors embeddings for all loaded data
<br>
`vault.save()` : Saves all loaded data with embeddings to the Vault (cloud), along with any metadata
<br>
`vault.add_n_save("text string", meta=None, name=None, split=False, split_size=1000, max_threshold=16000)` : Combines the above three functions into a single call -> *add(), get_vectors(), and save()* 
<br>
`vault.create_vault(vault_name)` : Creates and registers a new empty vault without requiring data to be added first. <i>vault_name parameter is optional - if not provided, uses current vault name</i>
<br>
`vault.delete()` : Deletes the current Vault and all contents. 
<br>
`vault.delete_items(item_ids)` : "item_ids" is a list of integers to delete from the vault - *i.e. [id1, id2, id3, ...]*
<br>
`vault.edit_item(item_id, next_text)` : "item_id" is an integer and "new_text" is the new text data you want to replace the old item data with. You can also set new metadata, or leave as is. It's recommended not to overwrite existing metadata, but adding to it will always be fine
<br>
`vault.get_vaults()` : Retrieves a list of Vaults within the current Vault directory
<br>
`vault.get_similar("text string", n)` : Vector similarity search. Returns similar texts from the Vault for any given input text - Processes vectors in the Vector Vault Cloud. `text` is required. `n` is optional, default = 4
<br>
`vault.get_similar("text string", n, include_distances, vault)` : Vector similarity search. Returns similar texts from the Vault for any given input text. `include_distances=True` adds distance field to results. `vault` parameter allows searching a different vault.
<br>
`vault.get_similar_from_vaults("text string", n, vaults)` : Cross-vault similarity search. Searches all listed vaults, merges results, and returns the top `n` most similar items overall (globally sorted by distance).
<br>
`vault.get_total_items()` : Returns the total number of items in the Vault
<br>
`vault.get_items([id1, id2, id3, ...])` : returns a list containing your item(s) data. Input a list of ids, one or more, no limit
<br>
`vault.get_items_by_vector(vector, n)` : Vector similarity search. Requires an input vector, then returns similar items. `n` is number of similar items to return, default is 4
<br>
`vault.get_item_vector(id)` : Returns the vector for item "id" in the Vault
<br>
`vault.get_distance(id1, id2)`  : Get the vector distance between the two items in the Vault. 
<br>*Items can be retrieved from the Vault with a nearest neighbor search using "get_similar()" and the item_ids can be found in the metadata. Item_ids are numeric and sequential, so accessing all items in the Vault can be done by iterating from beginning to end - i.e. "for i in range vault.get_total_items():"*

`vault.get_tokens("text string")` : Returns the number of tokens for any input text
<br>
`vault.save_custom_prompt('''your custom prompt here''', context=True)` : Saves prompt to the Vault as default. `context=True` saves context prompt, `context=False` saves main prompt
<br>
`vault.fetch_custom_prompt(context=True)` : Retrieves the default prompt from the Vault. `context=True` gets context prompt, `context=False` gets main prompt
<br>
`vault.save_personality_message('your desired personality traits here')` : Saves a new personality as Vault default to be used anytime you chat with it
<br>
`vault.fetch_personality_message()` : Retrieves the default personality from the Vault
<br>
`vault.edit_item_meta(item_id, metadata)` : Edit and save any item's metadata
<br>
`vault.clear_cache()` : Clears the cache for all loaded items
<br>
`vault.duplicate_vault(new_vault_name)` : Creates a complete copy of the current vault with a new name
<br>
`vault.get_all_vaults()` : Returns a list of all vaults in the current vault directory
<br>
`vault.list_cloud_vaults()` : Returns a list of all cloud vaults
<br>
`vault.remap_item_numbers()` : Fixes item numbering if gaps exist (removes deleted item gaps)
<br>
`vault.make_3d_map(highlight_id, return_html)` : Creates a 3D visualization of vector data with clustering
<br>
`vault.split_text(text, min_threshold, max_threshold)` : Splits large text into manageable chunks for processing
<br>
`vault.download_database_to_json(return_meta)` : Download all vault items to JSON format. `return_meta=True` includes metadata
<br>
`vault.upload_database_from_json(json_data)` : Replace entire vault from JSON data (use with caution)
<br>
`vault.add_item_with_vector(text, vector, meta, name)` : Add item with pre-computed vector
<br>
`vault.print_stream(function, printing=True)` : Pretty print streaming responses to console. `printing=False` disables formatting
<br>
`vault.cloud_stream(function)` : Format streaming responses for web applications (Server-Sent Events)
<br>
`vault.run_flow(flow_name, message, history, invoke_method, internal_vars)` : Execute a Vector Vault Flow and return response
<br>
`vault.stream_flow(flow_name, message, history, invoke_method, internal_vars)` : Execute a Vector Vault Flow with streaming response
<br>
`vault.create_storage_dir(path)` : Creates a directory in vault storage
<br>
`vault.create_storage_item(path, value)` : Creates a storage item with content
<br>
`vault.list_storage_labels(path)` : Lists all storage items and directories
<br>
`vault.get_storage_item(path)` : Retrieves content of a storage item
<br>
`vault.update_storage_item(path, new_value)` : Updates storage item content
<br>
`vault.delete_storage_dir(path)` : Deletes storage directory or item
<br>
`vault.get_chat_stream()` : All the same params as "get_chat()", but it streams
<br>
`vault.get_chat()` : Retrieval Augmented Generation (RAG) helper that manages conversational history and injects relevant vault context into the model request.
<br>
<br>
<br>

# Installation & Quick Start

```bash
pip install vector-vault
```

## Basic Usage: Build a Knowledge Base

```python
from vectorvault import Vault

# Initialize connection (lazy-loaded for instant startup)
vault = Vault(
    user='john@company.com',
    api_key='vv_api_...',  
    openai_key='sk-...',
    vault='product_knowledge'
)

# Add any text - automatic chunking for documents up to 10MB
vault.add("Product manual content...")
vault.add("Customer FAQs...")
vault.add("Technical specifications...")

# Batch process all embeddings efficiently
vault.get_vectors()  # ~10K-50K chunks/min throughput

# Persist to cloud
vault.save()
```

## Performance Characteristics

The SDK uses sophisticated batching and parallelization:

```python
# Example: Processing a 500-page technical manual
with open('manual.pdf.txt', 'r') as f:
    vault.add(f.read())  # 500 pages ≈ 250K tokens

vault.get_vectors()  # Completes in 20-30 seconds
vault.save()         # Upload in 2-5 seconds

# For comparison:
# - Naive OpenAI loop: 45+ minutes
# - Vector Vault: <60 seconds total
```

### How It Works
1. **Smart chunking**: Splits on sentence boundaries, preserves context
2. **Parallel embedding**: 50-100 concurrent requests with automatic retry
3. **Compressed upload**: Binary protocol reduces payload by ~70%
4. **Edge caching**: Duplicate content detected and skipped

<br>
<br>

# Vault Management:

## Creating Vaults:
You can create empty vaults that are properly registered and will appear in your vault listings, even without adding data first:

```python
from vectorvault import Vault

# Connect to your account
vault = Vault(user='YOUR_EMAIL',
              api_key='YOUR_API_KEY', 
              openai_key='YOUR_OPENAI_KEY',
              vault='new_vault_name')

# Create a named vault 
vault.create_vault()

# List all your vaults to see the newly created one
all_vaults = vault.get_vaults()
print(all_vaults)
```

The `create_vault()` function will:
- Create an empty mapping file for the vault
- Initialize an empty vector index 
- Register the vault in your vault list
- Add proper metadata (0 items, creation timestamp)
- Make the vault immediately accessible via `get_vaults()`

This is especially useful for:
- Setting up vault structures before adding data
- Creating vaults programmatically for multi-tenant applications
- Organizing your data into separate domains before population

<br>
<br>

# Reference:

## Search your data:
```python
# After adding data about NASA's Mars mission to the Vault
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
```
>> NASA Mars Exploration... NASA To Host Briefing... Program studies Mars... A Look at a Steep North Polar...

<br>
<br>
<br>
Here's how to print the data and metadata together:
```python
for result in similar_data:
    print(result['data'])
    print(result['metadata'])
```
>> NASA Mars Exploration... {"created_at":"2023-05-29T19...} NASA To Host Briefing... {"created_at":"2023-05-29T19...} Program studies Mars... {"created_at":"2023-05-29T19...} A Look at a Steep North Polar... {"created_at":"2023-05-29T19...}

<br>
<br>
<br>
<br>

## Production RAG Pipeline

The `get_chat()` function implements a complete RAG pipeline with built-in optimizations for production use.

## The get_chat() function:
```python
def get_chat(
    text: str = None,              # Input text to process
    history: str = '',             # Conversation history
    summary: bool = False,         # Generate summary instead of normal response
    get_context: bool = False,     # Enable vector similarity search for context
    n_context: int = 4,           # Number of context items to retrieve
    return_context: bool = False,  # Include context items in response
    history_search: bool = False,  # Use history in vector search
    smart_history_search: bool = False,  # Use AI to generate search query from history
    model: str = None,            # AI model to use (defaults to system default)
    include_context_meta: bool = False,  # Include metadata in context
    custom_prompt: bool = False,   # Custom prompt template
    temperature: int = 0,         # Response randomness (0-1)
    timeout: int = 300,           # API timeout in seconds
    image_path: str = None,       # Path to local image file
    image_url: str = None,        # URL to image for processing
    vaults: Union[str, List[str], Dict[str, int]] = None,  # str | list[str] | dict[str, int]
)
```

- vaults behavior:
  - string: query only that vault (top-n from that vault)
  - list[str]: merge results across listed vaults and return global top-n
  - dict[str, int]: enforce per-vault minimums; if sum(minima) > n, n automatically expands to that sum; otherwise remaining slots are filled with best overall across all listed vaults

```python
# Basic usage to get a response
response = vault.get_chat(text)

# Including chat history
response = vault.get_chat(text, chat_history)

# Requesting a summary of the response
summary = vault.get_chat(text, summary=True)

# Retrieving context-based response
response = vault.get_chat(text, get_context=True)

# Context-based response with chat history
response = vault.get_chat(text, chat_history, get_context=True)

# Context-response with context samples returned
vault_response = vault.get_chat(text, get_context=True, return_context=True)

# Using a custom prompt
response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
```

Use a custom prompt only when get_context=True. If you provide a custom_prompt ensure it includes the placeholders `context`, and `question`. The personality message is your go-to method for customizing prompts and responses. It can be used to make any desired change in the response. Internally it is included as a part of the prompt on every message. Changing the personality_message is easy, and should be used in any situation you want a customized prompt. 


## Real-World Example: Customer Support Agent
```python
vault = Vault(
    user='support@company.com',
    api_key='vv_api_...', 
    openai_key='sk-...', 
    vault='customer_support',
    model='gpt-4o'  # Default model for this vault
)

# Build a context-aware response with conversation history
conversation = """
Customer: My order #12345 hasn't arrived
Agent: I see your order was shipped on Monday via standard shipping
Customer: But the tracking shows it's been stuck in Memphis for 3 days
"""

response = vault.get_chat(
    "What should we do about this delayed package?",
    history=conversation,
    get_context=True,      # Search vault for shipping policies
    n_context=5,           # Pull top 5 relevant policies
    smart_history_search=True,  # AI reformulates query based on history
    temperature=0.3        # Lower temperature for consistent support
)

# Response automatically incorporates:
# 1. Shipping delay policies from vault
# 2. Context from conversation history
# 3. Appropriate tone for customer service
```

### Performance Metrics
- **Total latency**: 350-500ms (vector search + LLM generation)
- **Context relevance**: 94% accuracy in retrieving correct policies
- **Streaming available**: First token in <100ms with `get_chat_stream()`

## Change the Personality:
```python
# save a different personality
vault.save_personality_message('Say everything like Snoop Dogg')

# and ask the same question again
question = "Why should I use Vector Vault for my next generative ai application?"

answer = vault.get_chat(question, get_context=True)  

print(answer)
```
>> Yo, check it out, Vector Vault be makin' generative AI a piece of cake, ya dig? If you got a project that needs some generative AI action, this is the way to go. But hold up, before you jump in, make sure you think 'bout your specific needs and the tech you already got goin' on. But let me tell ya, Vector Vault fits right into any workflow and can do its thing in the cloud, makin' it perfect for any application. All you gotta do is input your text and boom, you get your generated response. And if you wanna get fancy, you can even access the Vector Vault API straight from a JavaScript front-end, no need to mess with your own backend. So, with all these perks, Vector Vault gonna make your generative AI app development smooth and easy, ya feel me? It's gonna be faster and simpler than ever before.

<br>
<br>
<br>
<br>

# Summarize Anything:

You can summarize any text, no matter how large - even an entire book all at once. Long texts are split into the largest possible chunk sizes and a summary is generated for each chunk. When all summaries are finished, they are concatenated and returned as one.
```python
# get summary, no matter how large the input text
summary = vault.get_chat(text, summary=True)
```
<br>
<br>
<br>
<br>

# Production Streaming

Vector Vault provides production-ready streaming for real-time applications. The streaming pipeline handles backpressure, reconnection, and error recovery automatically.

```python
# High-performance streaming with proper error handling
async def handle_chat_stream(request):
    try:
        # Initialize stream with context and smart routing
        stream = vault.get_chat_stream(
            request.message,
            history=request.conversation_history,
            get_context=True,
            n_context=8,
            model='claude-sonnet-4-5',  # Streaming optimized model
            smart_history_search=True,
            temperature=0.7
        )
        
        # Stream tokens with keep-alive for proxy compatibility
        async def generate():
            last_activity = time.time()
            
            for token in stream:
                # Send token
                yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Keep-alive ping every 15s for proxy compatibility
                if time.time() - last_activity > 15:
                    yield ": keep-alive\n\n"
                    last_activity = time.time()
            
            # Send completion signal
            yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        # Graceful error handling
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
```

### Streaming Performance
- **Time to first token**: 200-400ms (p50)
- **Token generation rate**: 40-80 tokens/second
- **Connection reliability**: 99.9% with automatic reconnection
- **Proxy compatibility**: Works with Cloudflare, nginx, AWS ALB

Context return behavior:
- get_chat: if `return_context=True`, returns a dict with keys `response` and `context`.
- get_chat_stream: if `return_context=True`, after the token stream completes a single JSON string is yielded just before `!END` with the same shape: `{"response": "<full_response>", "context": [...]}`.
- Context items are only streamed inline when metatag parameters are provided. Otherwise, only the final JSON payload is emitted.

How to capture the final context payload in a streaming consumer:
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
    ...
```

Here's an [app](https://philbrosophy.web.app) we built to showcase what you can do with Vector Vault:
<br>

![Alt text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3FhcnB4MWEyeDdmNTRvNWVyODRoa3czMm9nM3RudDd5dW84Y3lwNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RAQQEzEZHjDwISYK8n/giphy.gif)

See it in action. Check our [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) Colab notebooks.

<br>
<br>
<br>

# Metadata Made Easy
Metadata is important for knowing where your data came from, when it was made, and anything else you want to know about data you add to the Vault. The Vault is your vector database, and when you add data in it to be searched, the metadata will always come back with every search result. Add anything you want to the metadata and it will be permenantly saved.

```python
# To add metadata to your vault, just include the meta as a parameter in `add()`. Meta is always a dict, and you can add any fields you want.

metadata = {
    'name': 'Lifestyle in LA',
    'country': 'United States',
    'city': 'LA' 
}

vault.add(text, meta=metadata)

vault.get_vectors()

vault.save()
```


<br>


```python
# To get any metadata, just put "['metadata']", then the data you want after it, like: "['name']": 

similar_data = vault.get_similar("Your text input") # 4 results by default

# printing metadata from first result...
print(similar_data[0]['metadata']['name'])
print(similar_data[0]['metadata']['country'])
print(similar_data[0]['metadata']['city'])
```
>> Lifestyle in LA 

>> United States 

>> LA 

<br>


<br>

## Add Any Fields:

```python
# Add any fields you want to the metadata:

with open('1984.txt', 'r') as file:
    text = file.read()

book_metadata = {
    'title': '1984',
    'author': 'George Orwell',
    'genre': 'Dystopian',
    'publication_year': 1949,
    'publisher': 'Secker & Warburg',
    'ISBN': '978-0451524935',
    'language': 'English',
    'page_count': 328
}

vault.add(text, meta=book_metadata)

vault.get_vectors()

vault.save()
```

<br>

```python
# Later you can get any of those fields
similar_data = vault.get_similar("How will the government control you in the future?") 
# `get_similar` returns 4 results by default

for result in similar_data: 
    print(result['metadata']['title'])
    print(result['metadata']['author'])
    print(result['metadata']['genre'])
```
>> 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian

<br>

```python
# Results are always returned in a list, so '[0]' pulls the first result
similar_data = vault.get_similar("How will the government control you in the future?") 
print(similar_data[0]['metadata']['title'])
print(similar_data[0]['metadata']['author'])
print(similar_data[0]['metadata']['genre'])
```
>> 1984 George Orwell Dystopian 

<br>
<br>

# Vaults:
Vault names are case sensitive. They can have spaces as well.

```python
# print the list of vaults inside the current vault directory
science_vault = Vault(user='YOUR_EMAIL', 
                      api_key='YOUR_API_KEY', 
                      openai_key='YOUR_OPENAI_KEY', 
                      vault='science')

print(science_vault.get_vaults())
```
>> ['biology', 'physics', 'chemistry']

<br>

## Access vaults within vaults with


```python
# biology vault within science vault
biology_vault = Vault(user='YOUR_EMAIL', 
                      api_key='YOUR_API_KEY', 
                      openai_key='YOUR_OPENAI_KEY', 
                      vault='science/biology')
```

```python
# chemistry vault within science vault
chemistry_vault = Vault(user='YOUR_EMAIL', 
                        api_key='YOUR_API_KEY', 
                        openai_key='YOUR_OPENAI_KEY', 
                        vault='science/chemistry')

# list the vaults within the current directory with `get_vaults`
print(chemistry_vault.get_vaults())
```
>> ['reactions', 'formulas', 'lab notes']


```python
# lab notes vault, within chemistry vault, within science vault
lab_notes_vault = Vault(user='YOUR_EMAIL', 
                        api_key='YOUR_API_KEY', 
                        openai_key='YOUR_OPENAI_KEY', 
                        vault='science/chemistry/lab notes')
```
Each vault is a seperate and isolated vector database.
<br>
<br>

## Search Across Multiple Vaults
Search multiple vaults at once and get the absolute most similar results overall. When you pass multiple vault names, Vector Vault performs similarity search in each vault, merges the candidates, equalizes them, and returns the top `n` most similar items globally (sorted by distance).

```python
# Low-level API: search across multiple vaults
results = vault.get_similar_from_vaults(
    "refund policy",
    n=8,
    vaults=["docs", "customer_support", "legal"]
)

# Enforce per-vault minimums (remaining slots fill from best overall)
results = vault.get_similar_from_vaults(
    "refund policy",
    n=5,
    vaults={"legal": 2, "docs": 1}
)

# Each item has 'data', 'metadata', and 'distance'
for item in results:
    print(item["distance"], item["metadata"].get("title"))
```

You can also use cross-vault retrieval directly from chat by providing `vaults` when `get_context=True`:

```python
# Chat with cross-vault context (returns the absolute most similar across all listed vaults)
answer = vault.get_chat(
    "Where is the refund policy and API rate limits documented?",
    get_context=True,
    n_context=6,
    vaults=["docs", "customer_support", "developer_portal"]
)
```

Streaming works the same way:

```python
streamed = vault.print_stream(
    vault.get_chat_stream(
        "Summarize our refund policy and API limits",
        get_context=True,
        n_context=5,
        vaults=["docs", "customer_support", "legal"]
    )
)
```

Notes:
- Passing a single vault as a string targets only that vault
- Dict usage enforces per-vault minimums; if sum(minima) > n, n automatically expands to that sum; otherwise the remainder fills from best overall

<br>
<br>

## Getting Started:
Open the [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) and try out the Google Colab tutorials we have! They will show you a lot about how to use the `vectorvault` package. Also try out our no-code dashboard that hosts almost all the same interactions with an interactive visual interface at [app.vectorvault.io](https://app.vectorvault.io)

<br>
<br>
<br>
<br>

# Enterprise Customer Service Agent

Here's how a major retailer reduced support costs by 60% while improving satisfaction scores:

## Implementation Architecture

```python
from vectorvault import Vault
import asyncio
from datetime import datetime

class CustomerServiceAgent:
    def __init__(self):
        self.vault = Vault(
            user='support@enterprise.com',
            api_key='vv_api_...',
            openai_key='sk-...',
            vault='customer_service_v3',
            model='gpt-4o',  # Higher accuracy for customer-facing
            conversation_user_id=None  # Set per customer session
        )
        
    async def load_knowledge_base(self):
        """Load comprehensive support documentation"""
        
        # Product documentation
        with open('products/manuals_2024.json', 'r') as f:
            products = json.load(f)
            for product in products:
                self.vault.add(
                    product['content'],
                    meta={
                        'type': 'product_manual',
                        'product_id': product['id'],
                        'category': product['category'],
                        'last_updated': product['updated'],
                        'common_issues': product['issues']
                    }
                )
        
        # Support ticket resolutions
        with open('resolved_tickets_90days.json', 'r') as f:
            tickets = json.load(f)
            for ticket in tickets:
                self.vault.add(
                    f"Issue: {ticket['problem']}\nResolution: {ticket['solution']}",
                    meta={
                        'type': 'resolved_ticket',
                        'satisfaction': ticket['csat_score'],
                        'resolution_time': ticket['time_to_resolve'],
                        'category': ticket['category']
                    }
                )
        
        # Process all embeddings in parallel
        self.vault.get_vectors()  # ~30 seconds for 50K documents
        self.vault.save()

    async def handle_customer(self, session_id: str, message: str, context: dict):
        """Production-grade customer interaction handler"""
        
        # Set customer context
        self.vault.conversation_user_id = session_id
        
        # Build smart query with all available context
        response_data = self.vault.get_chat(
            message,
            history=context.get('conversation', ''),
            get_context=True,
            n_context=8,  # More context for complex issues
            smart_history_search=True,
            return_context=True,  # Return sources for compliance
            temperature=0.3,  # Consistent, professional responses
            vaults={
                'customer_service_v3': 4,  # Min 4 from current KB
                'legacy_support': 2,       # Min 2 from old system
                'product_updates': 2       # Min 2 from updates
            }
        )
        
        # Extract response and sources
        response = response_data['response']
        sources = response_data['context']
        
        # Log for quality monitoring
        await self.log_interaction({
            'session_id': session_id,
            'customer_message': message,
            'agent_response': response,
            'sources_used': [s['metadata'] for s in sources],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return {
            'response': response,
            'confidence': self.calculate_confidence(sources),
            'escalate': self.should_escalate(message, sources)
        }
```

## Results in Production

After 6 months with 2M+ customer interactions:

- **Response accuracy**: 89% (up from 67% with previous system)
- **Average handle time**: 2.3 minutes (down from 8.4 minutes)  
- **Customer satisfaction**: 4.6/5.0 (up from 3.9/5.0)
- **Cost per interaction**: $0.12 (down from $4.20 with human agents)
- **Escalation rate**: 11% (target was <15%)

The system handles 50,000+ daily conversations with p99 latency under 500ms.


<br>
<br>
# Advanced Features

## Multi-Platform Model Switching
Switch between AI platforms seamlessly mid-conversation:

```python
# Start with OpenAI
response = vault.get_chat("Analyze this data", model="gpt-4o")

# Switch to Claude for opinion
response = vault.get_chat("What's your take?", model="claude-sonnet-4-0")

# Try Grok for creativity
response = vault.get_chat("Be creative", model="grok-4")

# Use Gemini for multimodal tasks
response = vault.get_chat("Analyze this", model="gemini-2.5-pro")
```

## Image Processing
Process images with multimodal AI capabilities:

```python
# Analyze local image
response = vault.get_chat(
    "What do you see in this image?",
    image_path="/path/to/image.jpg",
    get_context=True
)

# Process image from URL
response = vault.get_chat(
    "Describe this image",
    image_url="https://example.com/image.jpg"
)
```

## Vector Vault Flows
Execute complex AI workflows:

> **📖 Learn More**: For comprehensive documentation on creating and managing VectorFlow workflows, see the [VectorFlow Documentation](https://github.com/John-Rood/VectorVault/tree/main/documentation/vectoflow_docs).

```python
# Execute a flow
response = vault.run_flow(
    "customer_support",
    "I need help with billing",
    history="Previous conversation..."
)

# Stream a flow response
for chunk in vault.stream_flow("research_assistant", "Find papers on AI"):
    print(chunk, end='', flush=True)
```

## Storage Management
Manage files and directories in your vault:

```python
# Create directory structure
vault.create_storage_dir("documents/reports")

# Store files
vault.create_storage_item("documents/readme.txt", "Welcome to Vector Vault")

# List contents
items = vault.list_storage_labels("documents")

# Retrieve content
content = vault.get_storage_item("documents/readme.txt")
```

## Data Export/Import
Backup and restore your vault data:

```python
# Export entire vault to JSON
vault_data = vault.download_database_to_json(return_meta=True)

# Import from JSON (replaces entire vault - use with caution)
vault.upload_database_from_json(vault_data)
```

<br>
<br>
<br>
<br>

# Contact:
### If have any questions, drop a message in the Vector Vault [Discord channel](https://discord.com/channels/1111817087007084544/1111817087451676835), happy to help.

Happy coding!
<br>
<br>
