![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a cloud-native vector database combined with OpenAI. Easily call ChatGPT or GPT4 and customize how they respond. Take any text data, vectorize it, and add it to the cloud vector database in 3 lines of code. Vector Vault enables you to quickly and securely create and interact with your vector databases - aka "Vaults". Vaults are hosted on serverless distributed cloud architecture backed by Google, making `vectorvault` scalable to any project size. 

`vectorvault` takes inspiration from LangChain, integrating their most popular chat features and LLM tools. However, by combining vector databases with OpenAI's chat into one single package, `vectorvault` is able to hide most of the complexity, making it simple to build custom chat experiences. It's even easier to use ChatGPT with the `vectorvault` package than OpenAI's default package, and you can customize what ChatGPT says by adding the kind of things you want it to say to the Vault. 

See tutorials in the Examples folder. You don't need a Vector Vault API key to use the tools or chat features, but you will need one to access the Vault Cloud and create/use vector databases. If you don't already have one, you can sign up for a free at [VectorVault.io](https://vectorvault.io). While the service is paid at production scale, the first tier is free allowing you to develop quickly. Affordability is the best reason to use Vector Vault. Due to our serverless cloud architecture, you are able to create an unlimited number of isolated vetor databases, while only paying for the number of references you make to them. Vector Vault was created in response to the need to have one vector database per client, while having many clients. 

<br>

### Full Python API:

`pip install vector-vault` : install
<br>
`from vectorvault import Vault` : import
<br>
`v = Vault(
  user='your_eamil', 
  api_key='your_api_key',
  openai_key='your_openai_api_key')` Create Vault Instance and Connect to OpenAI. *(Also call `verbose=True` to print all communications and notifications to the terminal while building)*
<br>
`v.add(text, meta=None, name='', split=False, split_size=1000)` : Loads data to be added to the Vault, with automatic text splitting for long texts. `text` is a text string. `meta` is a dictionary. *(`split=True` will split your text input, based on your `split_size`. For each split, a new item will be created. `name` parameter is a shortcut to adding a "name" field to the metadata. If you don't add a name or any metadata, generic info will be added for you. `text` is the only required input)*
<br>
`v.get_vectors()` : Retrieves vectors embeddings for all loaded data. *(No parameters)*
<br>
`v.save()` : Saves all loaded data with embeddings to the Vault (cloud), along with any metadata. *(No parameters)*
<br>
`v.delete()` : Deletes the current Vault and all contents. *(No parameters)*
<br>
`v.get_vaults()` : Retrieves a list of Vaults within the current Vault directory. *(No parameters)*
<br>
`v.get_similar(text, n)` : Retrieves similar texts from the Vault for a given input text - Processes vectors in the cloud. `text` is required. `n` is optional, default = 4.
<br>
`v.get_similar_local(text, n)` : Retrieves similar texts from the Vault for a given input text - Processes vectors locally. `text` is required. `n` is optional, default = 4. Local version for speed optimized local production.
<br>
`v.get_total_items()` : Returns the total number of items in the Vault
<br>
`v.clear_cache()` : Clears the cache for all the loaded items - *`add()` loads an item*
<br>
`v.get_items_by_vector(vector, n)` : Returns vector similar items. Requires input vector, returns similar items. `n` is number of items you want returned, default = 4
<br>
`v.get_distance(id1, id2)`  : For getting the vector distance between two items `id1` and `id2` in the Vault. 
<br>*Items can be retrieved from the Vault with a nearest neighbor search using `get_similar()` and the item_ids can be found in the metadata. Item_ids are numeric and sequential, so accessing all items in the Vault can be done by iterating from beginning to end - e.g. `for i in range vault.get_total_items():`*

`v.get_item_vector(id)` : returns the vector for item `id` in the Vault.
<br>
`v.get_items(ids)` : returns a list containing your item(s). `ids` is a list of ids, one or many
<br>
`v.cloud_stream(function)` : For cloud application yielding the chat stream, like a flask app. Called like *`v.cloud_stream(v.get_chat_stream('some_text'))`* in the return of a flask app.
<br>
`v.print_stream(function)` : For locally printing the chat stream. Called like *`v.print_stream(v.get_chat_stream('some_text'))`*. You can also assign a variable to it like *`reply = v.print_stream()`*  It still streams to the console, but the final complete text will also be available in the *`reply`* variable.
<br>
`v.get_chat()` : Retrieves a response from ChatGPT, with parameters for handling conversation history, summarizing responses, and retrieving context-based responses that reference similar data in the vault. *(See dedicated section below on using this function and its' parameters)*
<br>
`v.get_chat_stream()` : Retrieves a response from ChatGPT in stream format, with parameters for handling conversation history, summarizing responses, and retrieving context-based responses that reference similar data in the Vault. *(See dedicated section below on using this function and its' parameters)*
<br>
<br>
<br>

### LLM Exclusive Tools (`v.tools`):
• `get_rating`:
 Useful to get a quality rating
<br>
• `get_yes_no`:
 Useful for getting a difinitive answer 
<br>
• `get_binary`:
 Useful for getting a definitive answer in 0/1 format
<br>
• `get_match`:
 Useful to get an exact match to a single option within a set of options -> in: (text and list of answers) -> out: (exact match to one answer in list of answer)
<br>
• `get_topic`:
 Useful to classify the topic of conversation
<br>
• `match_or_make`:
 Get a match to a list of options, or make a new one if unrelated
 Useful if you aren't sure if the input will match one of your existing list options, and need flexibility of creating a new one. When starting from an empty list. - will create it from scratch

```python
# Tools example 1:
number_out_of_ten = v.tools.get_rating('how does LeBron James compare to Michael Jordan')

# Tools example 2: 
this_or_that = v.tools.get_binary('should I turn right or left, 0 for right, 1 for left')

# Tools example 3: 
answer = v.tools.get_yes_no('should I use Vector Vault to build my next AI project?')

print(answer)
```
>> yes



<br>
<br>
<br>

# Access The Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/2acebcaa-f5dd-44c9-8bba-c10723bc7064/Vector+Vault+Vault+2000.png" width="60%" height="60%" />
</p>

Install Vector Vault:
```
pip install vector-vault
```
<br>

# Build The Vault:

Set your openai key as an envorionment variable
```python
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
```

1. Create a Vault instance 
2. Gather some text data we want to store
3. Add the data to the Vault
4. Get vectors embeddings 
5. Save to the Vault Cloud

```python
from vectorvault import Vault

vault = Vault(user='YOUR_EMAIL', api_key='YOU_API_KEY', vault='NAME_OF_VAULT') 
# a new vault will be created if the name does not already exist 
# so you can create a Vault or connect to an exisiting Vault
# by calling this Vault instance

text_data = 'some data'

vault.add(text_data)

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
>> A test was done adding the full text of 37 books at once. The `get_vectors()` function took 8 minutes and 56 seconds. (For comparison, processing one at a time via OpenAI's embedding function would take roughly two days)

<br>
<br>

# Use The Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/5ae905b0-43d0-4b86-a965-5b447ee8c7de/Vector+Vault+Vault.jpg?content-type=image%2Fjpeg" width="60%" height="60%" />
</p>

From command line:
```
curl -X POST "https://api.vectorvault.io/get_similar" \
     -H "Content-Type: application/json" \
     -d '{
        "user": "your_username",
        "api_key": "your_api_key",
        "vault": "your_vault_name",
        "text": "Your text input"
     }'
```
>> [{"data":"NASA Mars Exploration... *(shortend for brevity)*","metadata":{"created_at":"2023-05-29T19:21:20.846023","item_id":0,"name":"webdump-0","updated_at":"2023-05-29T19:21:20.846028"}}]
    
<br>

In Python:
```python
# The same exact call, but in Python:
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
```
>> NASA Mars Exploration... NASA To Host Briefing... Program studies Mars... A Look at a Steep North Polar...

<br>
    
The metadata:
```python
print(similar_data[0]['metadata']) # printing from only the first result 
```
>> {"created_at":"2023-05-29T19:21:20.846023","item_id":0,"name":"webdump-0","updated_at":"2023-05-29T19:21:20.846028"}

<br>

Printing the data and metadata together:
```python
for result in similar_data:
    print(result['data'])
    print(result['metadata'])
```
>> NASA Mars Exploration... {"created_at":"2023-05-29T19...} NASA To Host Briefing... {"created_at":"2023-05-29T19...} Program studies Mars... {"created_at":"2023-05-29T19...} A Look at a Steep North Polar... {"created_at":"2023-05-29T19...}

<br>
<br>

# Metadata Made Easy


```python
# To add metadata to your vault, just include the meta as a parameter in `add()`. Meta is always a dict, and you can add any fields you want.

meta = {
    'name': 'Lifestyle in LA',
    'country': 'United States',
    'city': 'LA' 
}

vault.add(text, meta)

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

## Any Fields:

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

vault.add(text, book_metadata)

vault.get_vectors()

vault.save()
```

<br>

```python
# Later you can get all those fields
similar_data = vault.get_similar("How will the government control you in the future?") 

for result in similar_data:
    print(result['metadata']['title'])
    print(result['metadata']['author'])
    print(result['metadata']['genre'])
```
>> 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian

<br>

```python
# list is always returned, so you can do a for loop like above or numerically like this
similar_data = vault.get_similar("How will the government control you in the future?") 
print(similar_data[0]['metadata']['title'])
```
>> 1984

<br>
<br>

# Change Vaults

```python
# print the list of vaults inside the current vault directory
science_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science')

print(science_vault.get_vaults())
```
>> ['biology', 'physics', 'chemistry']

<br>

## Access vaults within vaults with


```python
# biology vault within science vault
biology_vault = Vault(user='YOUR_EMAIL', api_key='YOUR_API_KEY', vault='science/biology')
```

```python
# chemistry vault within science vault
chemistry_vault = Vault(user='YOUR_EMAIL', api_key='YOUR_API_KEY', vault='science/chemistry')

print(chemistry_vault.get_vaults())
```
>> ['reactions', 'formulas', 'lab notes']


```python
# lab notes vault within chemistry vault
lab_notes_vault = Vault(user='YOUR_EMAIL', api_key='YOUR_API_KEY', vault='science/chemistry/lab notes')
```

<br>
<br>

### Use `get_chat()` with `get_context=True` to get response from chatgpt referencing vault data

```python
question = "Should I use Vector Vault for my next generative ai application?"

answer = vault.get_chat(question, get_context=True)  

print(answer)
```
>> Vector Vault makes building generative ai easy, so you should consider using Vector Vault for your next generative ai project. Additionally, it is important to keep in mind your specific use cases and the other technologies you are working with. However, given the fact that Vector Vault can be integrated in any work flow and be isolated in a cloud environment, it is an ideal package to integrate into any application that you want to utilize generative ai with. To do so, just send the text inputs to your Vector Vault implementation and return the response. With this in mind, it is likely that Vector Vault would make building your next generative ai application both faster and easier.

<br>

To integrate vault data in the response, you need to pass `get_context=True` 
```python
# this will get context from the vault, then ask chatgpt the question
answer = vault.get_chat(question, get_context=True) 

# this will send to chatgpt only and not interact with the Vault in any way
answer = vault.get_chat(question) 
```


<br>
<br>

# ChatGPT
## Use ChatGPT with `get_chat()` 

<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/74776e31-4bfd-4d6b-837b-674790ca4288/wisdomandwealth_Electric_Yellow_and_Dark_Blue_-_chat_messages_g_c81a4325-5347-44a7-879d-a58a6d115446.png" width="60%" height="60%" />
</p>
<br>

Get chat response from OpenAI's ChatGPT. 
Rate limiting, auto retries, and chat histroy slicing auto-built-in so you can create complex chat capability without getting complicated. 
Enter your text, optionally add chat history, and optionally choose a summary response instead (default: summmary=False)

<br>
<br>

## The get_chat() function:
`get_chat(self, text: str, history: str = None, summary: bool = False, get_context = False, n_context = 4, return_context = False, history_search = False, model='gpt-3.5-turbo', include_context_meta=False, custom_prompt=False)`

- Example Signle Usage: 
`response = vault.get_chat(text)`

- Example Chat: 
`response = vault.get_chat(text, chat_history)`

- Example Summary: 
`summary = vault.get_chat(text, summary=True)`

- Example Context-Based Response:
`response = vault.get_chat(text, get_context=True)`

- Example Context-Based Response w/ Chat History:
`response = vault.get_chat(text, chat_history, get_context=True)`

- Example Context-Response with Context Samples Returned:
`vault_response = vault.get_chat(text, get_context=True, return_context=True)`
Response is a string, unless return_context == True, then response will be a dictionary 

- Example Custom Prompt:
`response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

`custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
`llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. 

*Example with GPT4:*

```python
response = vault.get_chat(text, chat_history, get_context=True, model='gpt4)
```

Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
`llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

*Example Custom Prompt:*  
```python
# You can build a custom prompt with custom variables:
my_prompt = """
    Use the following information to answer the Question at the end. 

    Math Result: {math_answer}

    Known Variables: {known_vars}

    Question: {question}

    (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
    Answer:
""" 
response = vault.get_chat(custom_prompt=my_prompt)
```
A custom prompt makes the get_chat() function flexible for any use case. Check ai.py to see the stock prompt templates, and get a better idea of how they work...or just send me a message in Discord.

<br>


## Normal Usage:
```python
# connect to the vault you want to use
vault = Vault(user='YOUR_EMAIL', api_key='YOUR_API_KEY', vault='philosophy')

# text input
question = "How do you find happiness?"

# get response
answer = vault.get_chat(question, get_context=True)

print(answer)
```
>> The answer to finding happiness is not one-size-fits-all, as it can mean different things to different people. However, it has been found that happiness comes from living and working in line with your values and virtues, and finding pleasure in the actions that accord with them. Additionally, having good friends who share your values and provide support and companionship enhances happiness. It is important to remember that happiness cannot be solely dependent on external factors such as material possessions or fleeting pleasures, as they are subject to change and instability. Rather, true happiness may come from an inner sense of mastery and control over yourself and your actions, as well as a sense of purpose and meaning in life.

<br>
<br>

# Summarize Anything:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/e1ff4ca3-e18b-4c8f-b3c9-ff6ddcc907a1/wisdomandwealth_a_summary_being_created._A_bunch_of_texts_are_f_df58744a-13cb-46fd-b39d-3f090349bbb7.png" width="60%" height="60%" />
</p>

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

# Streaming:
Use the built-in streaming functionality to get interactive chat streaming. Here's an [app](https://philbrosophy.web.app) we built to showcase what you can do with Vector Vault:
<br>

![Alt text](https://github.com/John-Rood/VectorVault/blob/778c11dfc8b71675d704c5f559c3452dc65b910a/digital%20assets/Streaming%20Demo%20Offish.gif)

## get_chat_stream():
See it in action. Check our [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) that has Colab notebooks you can be running in the browser seconds from now.

The `get_chat()` function returns the whole message at once, whereas the `get_chat_stream()` yields each word as it's received. Other than that, they are nearly identical and have nearly the same input parameters. Streaming is a much better experience and the preferred option for front end applications users interact with.

```python
## get_chat()
print(vault.get_chat(text, history))

## get_chat_stream()
for word in vault.get_chat_stream(text, history):
        print(word)
```

```python
# But it's best to use the built in print function: print_stream() 
vault.print_stream(vault.get_chat_stream(text, history))
```

```python
# With print_stream() final answer is returned after streaming completes, so you can make it a variable
answer = vault.print_stream(vault.get_chat_stream(text, history))
```

<br>
<br>

## The get_chat_stream() function:
`get_chat_stream(self, text: str, history: str = None, summary: bool = False, get_context = False, n_context = 4, return_context = False, history_search = False, model='gpt-3.5-turbo', include_context_meta=False, metatag=False, metatag_prefixes=False, metatag_suffixes=False, custom_prompt=False)`

Always use this get_chat_stream() wrapped by either print_stream(), or cloud_stream().
cloud_stream() is for cloud functions, like a flask app serving a front end elsewhere.
print_stream() is for local console printing

- Example Signle Usage: 
`response = vault.print_stream(vault.get_chat_stream(text))`

- Example Chat: 
`response = vault.print_stream(vault.get_chat_stream(text, chat_history))`

- Example Summary: 
`summary = vault.print_stream(vault.get_chat_stream(text, summary=True))`

- Example Context-Based Response:
`response = vault.print_stream(vault.get_chat_stream(text, get_context = True))`

- Example Context-Based Response w/ Chat History:
`response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True))`

- Example Context-Response with Context Samples Returned:
`vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True))`

- Example Custom Prompt:
`response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)`

`custom_prompt` overrides the stock prompt we provide. Check ai.py to see the originals we provide. 
`llm` and `llm_stream` models manage history internally, so the `content` is the only variable to be included and formattable in the prompt. Visit the get_chat_stream() function in vault.py for more information on metatags or check out our examples folder streaming tutorial.

*Example with GPT4:*

```python
response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, model='gpt4))
```

Getting context from the Vault is usually the goal when customizing text generation, and doing that requires additional prompt variables.
`llm_w_context` and `llm__w_context_stream` models inject the history, context, and user input all in one prompt. In this case, your custom prompt needs to have `history`, `context` and `question` formattable in the prompt like so:

*Example with Custom Prompt:*  
```python
my_prompt = """
    Use the following Context to answer the Question at the end. 
    Answer as if you were the modern voice of the context, without referencing the context or mentioning that fact any context has been given. Make sure to not just repeat what is referenced. Don't preface or give any warnings at the end.

    Chat History (if any): {history}

    Additional Context: {context}

    Question: {question}

    (Respond to the Question directly. Be the voice of the context, and most importantly: be interesting, engaging, and helpful) 
    Answer:
""" 
response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True, custom_prompt=my_prompt))
```

<br>

Streaming is a key for front end applications, so we also built a `cloud_stream` function to make cloud streaming to your front end app easy. In a flask app, all you need to do is recieve the customer text, then call the vault in the return like this: 
```python
# Stream from a flask app in one line
return Response(vault.cloud_stream(vault.get_chat_stream(text, history, get_context=True)), mimetype='text/event-stream')
```
This makes going live with a high level app extremely fast and easy, plus your infrastructure will be scalable and robust. Now you can build impressive applications in record time! If have any questions, message in [Discord](https://discord.gg/AkMsP9Uq). Check out our Colab notebooks in the [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) you can run in the browser right now.


<br>
<br> 



<br>
<br>
<br>

# Build an AI Cusomter Service Chat Bot
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/dceb5c7d-6ec6-4eda-82f2-b8848c7b519d/ai_chatbot_having_a_conversation.png" width="60%" height="60%" />
</p>
<br>

In the following code, we will add all of a company's past support conversations to a cloud Vault. (We load the company support texts from a .txt file, vectorize them, then add them to the Vault). As new people message in, we will vector search the Vault for similar questions and answers. We take the past answers returned from the Vault and instruct ChatGPT to use those previous answers to answer this new question. (NOTE: This will also work based on a customer FAQ, or customer support response templates).

<br>

### Create the Customer Service Vault
```python
from vectorvault import Vault

os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

vault = Vault(user='your_email', api_key='your_api_key', vault='Customer Service')

with open('customer_service.txt', 'r') as f:
    vault.add(f.read())

vault.get_vectors()

vault.save()
```

<br>

And just like that, in a only a few lines of code we created a customer service vault. Now whenever you want to use it in production, just use the `get_chat()` with `get_context=True`, which will take the customer's question, search the vault to find the most similar questions and answers, then have ChatGPT reply to the customer using that information.

```python
customer_question = "I just bought your XD2000 remote and I'm having trouble syncing it to my tv"

chatbot_answer = vault.get_chat(customer_question, get_context=True)
```
<br>

That's all it takes to create an AI customer service chatbot that responds as well as any human support rep!


<br>
<br>

---
<br>
<br>



## Getting Started:
Open the [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) and try out the Google Colab tutorials we have! They will show you a lot, plus they are in Google Colab, so no local set up required, just open them up and press play.

<br>
<br>

## Contact:
### If have any questions, drop a message in the Vector Vault [Discord channel](https://discord.gg/AkMsP9Uq), happy to help.

Happy coding!
<br>
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/7d1a596b-7560-446b-aa69-1827819d198b/Looking+out+with+hope+vector+vault.png" width="60%" height="60%" />
</p>

<br>
