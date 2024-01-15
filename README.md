![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a cloud-native vector database. This Python package combines our cloud vector database with OpenAI embeddings and API calls. Easily call ChatGPT or GPT4 and customize how they respond. Take any text data, vectorize it, and add it to the cloud vector database in 1 line of code. Vector Vault enables you to easily create and interact with your cloud vector databases - aka "Vaults". Our cloud is hosted on a serverless distributed cloud architecture backed by Google, making Vector Vault scalable to any project size. 

Vector Vault takes inspiration from LangChain by integrating their most popular features and tools into our package. However, by combining vector databases with OpenAI's API into a single package, `vectorvault` is able to keep the complexity in the background, making it simple and easy to build powerful chat experiences. It's even easier to use ChatGPT with `vectorvault` than with OpenAI's default package, plus you can customize what ChatGPT says by adding things you want it to know to the Vault, and save a custom personality message to make it sound just right. 

You will need a Vector Vault API key to create and interact with vector databases, so pick one up for free at [VectorVault.io](https://vectorvault.io). The first tier is free so you can start building right away. Vector Vault is the first vector database platfrom to go fully serverless. This architecture makes Vector Vault the most affordable and scalable cloud vector database platform in the world. Due to our serverless nature, you are able to create an unlimited number of vector databases, while only paying for the number of references you make to them. `This makes Vector Vault the only real solution for multi-tenet applications, where you need an isolated vector database for each user.`  No matter what tier you are on, you will always be able to create and access an infinite number of isolated vector databases. See tutorials in the Examples folder. 


<br>
<br>
<br>
<br>

# Full Python API:

Install:
<br>
`pip install vector-vault` 

Connect:
```python
from vectorvault import Vault

vault = Vault(user='your_eamil', 
              api_key='your_api_key',
              openai_key='your_openai_api_key',
              vault='any_vault_name')
``` 

`vault.add("text string")` : Loads data to be added to the Vault, with automatic text splitting for long texts
<br>
`vault.get_vectors()` : Retrieves vectors embeddings for all loaded data
<br>
`vault.save()` : Saves all loaded data with embeddings to the Vault (cloud), along with any metadata
<br>
`vault.add_n_save("text string")` : Combines the above three functions into a sinlge call -> *add(), get_vectors(), and save()* 
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
`vault.get_similar_local("text string", n)` : Vector similarity search. Returns similar texts from the Vault for any given input text - Processes vectors locally. *This local version is speed optimization for production deployments.*
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
`vault.save_custom_prompt('''your custom prompt here''')` : Saves prompt to the Vault as default. Whenever you call "get_chat()" with parameter "get_context=True", the custom prompt you saved will be used 
<br>
`vault.fetch_custom_prompt()` : Retrieves the default prompt from the Vault
<br>
`vault.save_personality('your desired personality traits here')` : Saves a new personality as Vault default to be used anytime you chat with it. Now, whenever you call "get_chat()", the personality you saved will be used in the response
<br>
`vault.fetch_personality()` : Retrieves the default personality from the Vault
<br>
`vault.get_chat_stream()` : All the same params as "get_chat()", but it streams
<br>
`vault.get_chat()` : A cutting-edge function designed for Retrieval Augmented Generation (RAG), enabling you to effortlessly manage conversational history and seamlessly integrate knowledge from the Vault for context-based responses 
<br>
<br>
<br>

# Install:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/2acebcaa-f5dd-44c9-8bba-c10723bc7064/Vector+Vault+Vault+2000.png" width="60%" height="60%" />
</p>

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

# Reference:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/5ae905b0-43d0-4b86-a965-5b447ee8c7de/Vector+Vault+Vault.jpg?content-type=image%2Fjpeg" width="60%" height="60%" />
</p>

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

The exact same call, but from command line:
```
curl -X POST "https://api.vectorvault.io/get_similar" \
     -H "Content-Type: application/json" \
     -d '{
        "user": "your_username",
        "api_key": "your_api_key",
        "openi_key": "your_openai_api_key",
        "vault": "your_vault_name",
        "text": "Your text input"
     }'
```
>> [{"data":"NASA Mars Exploration... *(shortend for brevity)*","metadata":{"created_at":"2023-05-29T19:21:20.846023","item_id":0,"name":"webdump-0","updated_at":"2023-05-29T19:21:20.846028"}}]
    
<br>

Back to Python, here's how to print the data and metadata together:
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
get_chat( 
        text = None, # the text to respond to
        history = None, # conversation history
        summary = False, # generate a summary of the text
        get_context = False, # retrieves vector similary search results as context to augment the response (RAG)
        n_context = 4, # how many items to retrieve and use as context 
        return_context = False, # returns the context items
        history_search = False, # integrates conversation history into the vector similarity search
        smart_history_search = False, # uses ChatGPT to generate a search query given the conversation history
        model = 'gpt-3.5-turbo', # ChatGPT by default - change to 'gpt-4' or any of the other models
        include_context_meta = False, # include item metadata in the RAG response
        custom_prompt = False, # a custom prompt to use in place of the default 
        local = False, # performs vector search locally
        temperature = 0, # randomness of the model 0 = none, 0.5 = mid, 1 = most
        timeout = 45 # how many seconds to wait when not receiving a response from the AI 
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


## Normal Usage:
```python
# connect to the vault you want to use
vault = Vault(user='YOUR_EMAIL', 
              api_key='YOUR_API_KEY', 
              openai_key='YOUR_OPENAI_KEY', 
              vault='vectorvault')

# text input
question = "Should I use Vector Vault for my next generative ai application?"

answer = vault.get_chat(question, get_context=True)  

print(answer)
```
>> Vector Vault simplifies the process of creating generative AI, making it a compelling choice for your next project involving generative AI. It's essential to consider your specific use cases and the technologies you're currently utilizing. Nevertheless, Vector Vault's seamless integration into various workflows and its ability to operate in a cloud-based environment make it an ideal solution for incorporating generative AI into any application. To achieve this, you can simply input your text into your Vector Vault implementation and retrieve the generated response. Additionally, you have the option to access the Vector Vault API directly from a JavaScript front-end interface, eliminating the need for setting up your own backend implementation. With these advantages in mind, Vector Vault is likely to streamline the development of your next generative AI application, making it faster and more straightforward.

## Changing the Personality:
```python
# save a different personality
vault.save_personality_message('Say everything like Snoop Dogg')

# and ask the same question again
question = "Should I use Vector Vault for my next generative ai application?"

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

Want to make it a certain length?
```python
# make a summary under a legnth of 1000 characters
summary = vault.get_chat(text, summary=True)

while len(summary) > 1000:
    summary = vault.get_chat(summary, summary=True)
```

<br>
<br>
<br>
<br>

# Streaming:
Use the built-in streaming functionality to get interactive chat streaming with `get_chat_stream()`. It has all the same params as `get_chat()`, but it streams.
Here's an [app](https://philbrosophy.web.app) we built to showcase what you can do with Vector Vault:
<br>

![Alt text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3FhcnB4MWEyeDdmNTRvNWVyODRoa3czMm9nM3RudDd5dW84Y3lwNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RAQQEzEZHjDwISYK8n/giphy.gif)

See it in action. Check our [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) Colab notebooks.

<br>
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
<br>
<br>
<br>


# LLM Tools (`vault.tools`):
### Vector Vault's arsenal of exclusive LLM tools that allow you to generate structured and dependable outputs from unstructured and variable data:

`get_rating`: Generate quality ratings (1-10)
<br>
`get_yes_no`: Generate a definitive 'yes' or 'no' response
<br>
`get_binary`: Generate binary (0/1) response
<br>
`get_match`: Generate an exact match to one item within a list
<br>
`get_topic`: Classify any text into a 1-3 word topic
<br>
`match_or_make`: Match or create new list options for flexibility, even from scratch

```python
# Tools example 1:
number_out_of_ten = vault.tools.get_rating('How does LeBron James compare to Michael Jordan?')
print(number_out_of_ten)
```
>> 8

```python
# Tools example 2: 
binary_answer = vault.tools.get_binary('Should I turn right or left?, 0 for right, 1 for left')
print(binary_answer)
```
>> 0

```python
# Tools example 3: 
yes_or_no = vault.tools.get_yes_no('Should I use Vector Vault to build my next AI project?')
print(yes_or_no)
```
>> yes



<br>
<br>
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
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/dceb5c7d-6ec6-4eda-82f2-b8848c7b519d/ai_chatbot_having_a_conversation.png" width="60%" height="60%" />
</p>
<br>

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
<br>
<br>
<br>
<br>

# Contact:
### If have any questions, drop a message in the Vector Vault [Discord channel](https://discord.gg/AkMsP9Uq), happy to help.

Happy coding!
<br>
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/7d1a596b-7560-446b-aa69-1827819d198b/Looking+out+with+hope+vector+vault.png" width="60%" height="60%" />
</p>

<br>
