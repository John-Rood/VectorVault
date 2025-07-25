![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)
# Vector Vault

Vector Vault is a cutting-edge, cloud-native and RAG-native vector database solution that revolutionizes the way developers integrate AI into their applications. Our platform seamlessly combines vector databases, similarity search, and AI model interactions into a single, easy-to-use service.

## Key Features

- **RAG-Native Architecture**: Perform Retrieval-Augmented Generation in one line of code.
- **Unparalleled Simplicity**: Implement sophisticated AI features with just a few lines of code.
- **Full-Stack Integration**: Use our Python package for backend operations and our JavaScript package for seamless front-end integration.
- **Vector Vault Cloud**: Our cloud service handles vector search, retrieval, and AI model interactions, simplifying your architecture.
- **One-Line Operations**: Save to the cloud vector database and generate RAG responses in one line of code. Easily customize responses and vector searches by adjusting parameters.
- **Developer-Centric Approach**: Focus on your application logic rather than complex AI and database integrations.
- **Unlimited Isolated Databases**: Create and access an infinite number of vector databases, ideal for multi-tenant applications.

## Advantages

- **Unmatched Ease of Use**: Integrate advanced AI features faster and more easily than with any other solution.
- **Front-End Integration**: Make vector calls directly from the front-end, a unique feature in the market.
- **Simplified Architecture**: Eliminate the need for complex backend setups for AI integration.
- **RAG Optimization**: Purpose-built to support and simplify Retrieval-Augmented Generation workflows.
- **Customization**: Tailor AI responses to your needs with minimal code.
- **Scalability**: Fully serverless platform offering unparalleled scalability.
- **Time and Resource Saving**: Dramatically reduce development time and resource allocation for AI feature integration.
- **Risk-Free Trial**: Start with a 30-day free trial to experience the full power of Vector Vault.

## Getting Started

1. Sign up for a 30-day free trial at [VectorVault.io](https://vectorvault.io) to get your API key.
2. Install the `vectorvault` package for JavaScript (frontend) or Python (backend).
3. Start building powerful AI-driven experiences with just a few lines of code.

## Use Cases

Vector Vault is particularly well-suited for:
- Rapid prototyping and development of AI-powered applications
- Projects requiring vector search and AI capabilities on the frontend
- Multi-tenant systems needing isolated vector databases for each user
- Applications demanding customized AI interactions with minimal backend complexity

## Learn More

Explore our documentation and examples to see how Vector Vault can revolutionize your development process. Start building with Vector Vault today and experience the future of simplified, RAG-native, cloud-native vector databases!
## Key Features

- **RAG-Native Architecture**: Perform Retrieval-Augmented Generation with minimal code, streamlining your AI-powered applications.
- **Cloud-Native Design**: Built on a serverless, distributed cloud infrastructure powered by Google, ensuring scalability for projects of any size.
- **Simplified AI Integration**: Easily interact with OpenAI's latest GPT, with the ability to customize responses in just a few lines of code.
- **Effortless Vectorization**: Convert any text data into vectors and add them to your cloud database with minimal effort.
- **Developer-Centric Approach**: Focus on building your application logic rather than wrestling with complex AI and database integrations.
- **Unlimited Isolated Databases**: Create and access an infinite number of vector databases, ideal for multi-tenant applications.
- **One Line**: Save to the cloud vector database in one line of code and generate RAG responses in one line of code. Edit the parameters of the call to change anything about the response or vector search. You have full control, and you get to exercise that control in just one line of code.
- **Multi-Platform AI Support**: Seamlessly switch between OpenAI, Claude, Grok, Groq, and Gemini models mid-conversation with automatic platform detection.
- **Advanced Streaming**: Real-time streaming responses with built-in utilities for console and web applications.
- **Smart History Search**: AI-powered contextual search query generation for multi-turn conversations.
- **Vector Vault Flows**: Execute complex AI workflows with `run_flow()` and `stream_flow()` functions.
- **Storage Management**: Built-in cloud storage system for managing files and directories.
- **Image Processing**: Support for multimodal AI interactions with image analysis capabilities.

## Getting Started

1. Sign up for a 30-day free trial at [VectorVault.io](https://vectorvault.io) to get your API key.
2. Install the `vectorvault` package: `pip install vector-vault`
3. Start building powerful AI-driven experiences with ease.

## Advantages

- **Simplicity**: Easier to use than traditional vector databases and AI integrations.
- **RAG Optimization**: Built from the ground up to support Retrieval-Augmented Generation workflows.
- **Customization**: Add specific knowledge to your Vault and tailor AI responses to your needs.
- **Scalability**: The first fully serverless, RAG-native vector database platform, offering unparalleled scalability.
- **Premium Service**: Vector Vault offers high-quality, professional-grade capabilities for demanding applications.
- **Risk-Free Trial**: Start with a 30-day free trial to experience the full power of Vector Vault before committing.
- **Efficiency**: Perform complex operations with minimal code, saving development time and reducing potential errors.

## Use Cases

Vector Vault is particularly well-suited for:
- AI-powered applications requiring quick and easy RAG implementation
- Multi-tenant systems needing isolated vector databases for each user
- Projects of any scale, from small experiments to large-scale deployments
- Applications demanding customized AI interactions with minimal complexity

## Learn More

Explore the Examples folder for tutorials and practical applications of Vector Vault.

Start building with Vector Vault today and experience the future of RAG-native, cloud-native vector databases!


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
              groq_key='your_groq_api_key',          # For Groq's fast inference
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
- **`groq_key`**: Get fast inference with Groq's optimized models
- **`grok_key`**: Access xAI's Grok models for creative tasks
- **`anthropic_key`**: Use Claude models for reasoning and analysis
- **`gemini_key`**: Access Google Gemini models for multimodal AI tasks

## Platform-Specific Models

```python
# OpenAI models (requires openai_key)
vault.get_chat("Hello", model="gpt-4o")
vault.get_chat("Hello", model="gpt-4o-mini") 
vault.get_chat("Hello", model="o1")

# Groq models (requires groq_key)
vault.get_chat("Hello", model="llama3-70b-8192")
vault.get_chat("Hello", model="mixtral-8x7b-32768")

# Grok models (requires grok_key)  
vault.get_chat("Hello", model="grok-4")
vault.get_chat("Hello", model="grok-3")

# Claude models (requires anthropic_key)
vault.get_chat("Hello", model="claude-sonnet-4-0")
vault.get_chat("Hello", model="claude-3-5-sonnet-20241022")

# Gemini models (requires gemini_key)
vault.get_chat("Hello", model="gemini-2.5-pro")
vault.get_chat("Hello", model="gemini-2.5-flash")
```

## Getting API Keys

- **Vector Vault**: Sign up at [vectorvault.io](https://vectorvault.io) for 30-day free trial
- **OpenAI**: Get your key at [platform.openai.com](https://platform.openai.com)
- **Groq**: Sign up at [console.groq.com](https://console.groq.com)  
- **xAI (Grok)**: Get access at [console.x.ai](https://console.x.ai)
- **Anthropic**: Apply for API access at [console.anthropic.com](https://console.anthropic.com)
- **Google Gemini**: Get your API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

## Advanced Initialization Options

```python
vault = Vault(
    user='your_email',
    api_key='your_vectorvault_api_key',
    openai_key='your_openai_key',
    groq_key='your_groq_key',        # Optional: For Groq models
    grok_key='your_grok_key',        # Optional: For Grok models
    anthropic_key='your_anthropic_key',  # Optional: For Claude models
    gemini_key='your_gemini_key',    # Optional: For Gemini models
    vault='vault_name',
    
    # Performance & Behavior
    verbose=True,                    # Enable detailed logging
    embeddings_model='text-embedding-3-large',  # Use larger embedding model
    
    # Conversation Features  
    conversation_user_id='user123',  # Enable conversation history
    model='claude-sonnet-4-0',       # Set default model
    
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
`vault.get_chat()` : A cutting-edge function designed for Retrieval Augmented Generation (RAG), enabling you to effortlessly manage conversational history and seamlessly integrate knowledge from the Vault for context-based responses 
<br>
<br>
<br>

# Install:
Install Vector Vault:
```
pip install vector-vault
```
<br>

# Upload:

```python
from vectorvault import Vault

vault = Vault(user='YOUR_EMAIL',
              api_key='YOUR_API_KEY', 
              openai_key='YOUR_OPENAI_KEY',
              vault='NAME_OF_VAULT') # a new vault will be created if the name does not exist - if so, you will be connected

vault.add('some text')

vault.get_vectors()

vault.save()
```


<br>
<br>

`vault.add()` is very versitile. You can add any length of text, even a full book...and it will be all automatically split and processed. `vault.get_vectors()` is also extremely flexible. You can `vault.add()` as much as you want, and then when you're done, process all the vectors at once with a single `vault.get_vectors()` call - Which internally batches vector embeddings with OpenAI's text-embeddings-ada-002, and comes with auto rate-limiting and concurrent requests for maximum processing speed. 
```python

vault.add(very_large_text)

vault.get_vectors() 

vault.save() 

# these three lines execute fast and can be called mid-conversation before a reply
```
Small save loads are usually finished in less than a second. Large loads depend on total data size. 
>> A 2000 page book (e.g. the Bible) would take ~30 seconds. A test was done adding 37 books. The `get_vectors()` function took 8 minutes and 56 seconds. (For comparison, processing via OpenAI's standard embedding function, that you can find in their documentation, would take over two days). This exponentially faster processing time is due to our built in concurrency and internal text uploading methods that are optimized for speed and have built-in rate limiting.

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

## Talk to your data

Get chat response from OpenAI's ChatGPT with `get_chat()`.
It has built-in rate limiting, auto retries, and automatic chat histroy slicing, so you can create complex chat capability without getting complicated. All you have to add is the text and the Vault takes care of the rest.

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
)
```

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


## Basic Usage:
```python
# connect to the vault you want to use
vault = Vault(user='YOUR_EMAIL', 
              api_key='YOUR_API_KEY', 
              openai_key='YOUR_OPENAI_KEY', 
              vault='vectorvault')

# text input
question = "Why should I use Vector Vault for my next generative ai application?"

answer = vault.get_chat(question, get_context=True)  

print(answer)
```
>> Vector Vault simplifies the process of creating generative AI, making it a compelling choice for your next project involving generative AI. It's essential to consider your specific use cases and the technologies you're currently utilizing. Nevertheless, Vector Vault's seamless integration into various workflows and its ability to operate in a cloud-based environment make it an ideal solution for incorporating generative AI into any application. To achieve this, you can simply input your text into your Vector Vault implementation and retrieve the generated response. Additionally, you have the option to access the Vector Vault API directly from a JavaScript front-end interface, eliminating the need for setting up your own backend implementation. With these advantages in mind, Vector Vault is likely to streamline the development of your next generative AI application, making it faster and more straightforward.

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

# Streaming:
Use the built-in streaming functionality to get interactive chat streaming with `get_chat_stream()`. It has all the same params as `get_chat()`, but it streams.
```python
# Console streaming with pretty printing
response = vault.print_stream(
    vault.get_chat_stream("Tell me about Vector Vault", get_context=True)
)

# Web application streaming (Server-Sent Events)
@app.route('/chat-stream')
def chat_stream():
    return Response(
        vault.cloud_stream(
            vault.get_chat_stream("User message", get_context=True)
        ),
        mimetype='text/event-stream'
    )
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
<br>
<br>




## Getting Started:
Open the [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) and try out the Google Colab tutorials we have! They will show you a lot about how to use the `vectorvault` package. Also try out our no-code dashboard that hosts almost all the same interactions with an interactive visual interface at [app.vectorvault.io](https://app.vectorvault.io)

<br>
<br>
<br>
<br>

# Build an AI Cusomter Service Chatbot
Here's a quick example of what you can do with Vector Vault. We load a company's customer support data into a txt file called `customer_service.txt`, vectorize all that data, then upload it to the Vault. 

<br>

### Create the Customer Service Vault
```python
from vectorvault import Vault

vault = Vault(user='your_eamil', 
              api_key='your_api_key',
              openai_key='your_openai_api_key',
              vault='Customer Service')

with open('customer_service.txt', 'r') as f:
    vault.add(f.read())

vault.get_vectors()

vault.save()
```

<br>

Now whenever you want to use it in production call `get_chat()`, with `get_context=True`, which will take the customer's question, search the Vault to find the 4 most relevant answers, then have ChatGPT reply to the customer using those answers to augment its' reply. AKA RAG response.

```python
customer_question = "I just bought your XD2000 remote and I'm having trouble syncing it to my tv"

support_answer = vault.get_chat(customer_question, get_context=True)
```
Now your AI chatbot sounds just like every other rep!


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
### If have any questions, drop a message in the Vector Vault [Discord channel](https://discord.gg/AkMsP9Uq), happy to help.

Happy coding!
<br>
<br>
