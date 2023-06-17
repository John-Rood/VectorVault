![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a vector database cloud service built to make generative ai chat quick and easy. It allows you to seamlessly vectorize data and access it from the cloud. It's scalable to both small projects and large applications with millions of users. Vector Vault has been designed with a user-friendly code interface to make the process of working with vector search easy and let you focus on what matters, results. Vector Vault ensures secure and isolated data handling and enables you to create and interact vector databases - aka "vaults" - in the cloud with under one second response times.

The `vectorvault` package comes with extensive chat functionality, so that you don't have to think about the details and can make smooth chat applications with ease. Speaking of smooth chat experiences, `vectorvault` also comes with streaming built-in, so  you can make customer-ready applications fast (see below).

Popular packages for building functionality with llms, like LangChain, can often be complicated and difficult when it comes to referencing vector databases, but referencing your vector database is crucial to create the chat experience you are going for. This is one of the reasons we built `vectorvault`... We've integrated all the chat options people like to use with LangChain, but made them all easier and more straight forward to use. Now with Vector Vault, integrating vector database results into generative chat applications is not only easy, it's the default. With `vectorvault` have total control over every aspect with parameters. If you have been looking for an easy and reliable way to use vector databases with ChatGPT, then Vector Vault is for you.

By combining vector similarity search with generative ai chat, new possibilities for conversation and communication emerge. For example, product information can be added to a Vault, and when a customer asks a product question, the right product information can be instantly retreived and seamlessly used in conversation by chatgpt for an accurate response. This capability allows for informed conversation and the possibilites range from ai automated customer support, to new ways to get news, to ai code reviews that reference source documentation, to ai domain experts for specific knowledge sets, and much more. You will need an api key in order to access the Vault Cloud. If you don't already have one, you can sign up at [VectorVault.io](https://vectorvault.io)

<br>

Basic Interactions:

`add()` : Prepares data to be added to the Vault, with automatic text splitting and processing for long texts. 
<br>
`get_vectors()` : Retrieves vectors embeddings for all prepared data 
<br>
`save()` : Saves the data with embeddings to the Vault (cloud), along with any metadata
<br>
`delete()` : Deletes the current Vault and all contents
<br>
`get_vaults()` : Retrieves a list of Vaults within the current Vault directory
<br>
`get_similar()` : Retrieves similar texts from the Vault for a given input text - We process vectors in the cloud
<br>
`get_similar_local()` : Retrieves similar texts from the Vault for a given input text - You process vectors locally
<br>
`get_chat()` : Retrieves a response from ChatGPT, with support for handling conversation history, summarizing responses, and retrieving context-based responses by referencing similar data in the vault

>> `get_vectors()` utilizes openai embeddings api and internally batches vector embeddings with OpenAI's text-embeddings-ada-002 model, and comes with auto rate-limiting and concurrent requests for maximum processing speed


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

### Get Your Vector Vault API Key:
```python
from vectorvault import register

register(first_name='FIRST_NAME', last_name='LAST_NAME', email='YOUR_EMAIL', password='make_a_password')
```
The api key will be sent to your email.

<br>

# Build The Vault:

Set your openai key as an envorionment variable
```python
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
```

1. Create a Vault instance - (new vault will be created if name does not exist)
2. Gather some text data we want to store
3. Add the data to the Vault
4. Get vectors embeddings 
5. Save to the cloud vault

```python
from vectorvault import Vault

vault = Vault(user='YOUR_EMAIL', api_key='YOU_API_KEY', vault='NAME_OF_VAULT)

text_data = 'some data'

vault.add(text_data)

vault.get_vectors()

vault.save()
```

<br>
<br>

Now that you have saved some data to the vault, you can add more at anytime. `vault.add()` is very versitile. You can add any length of text, even a full book...and it will be all automatically split and processed. `vault.get_vectors()` is also extremely flexible, because you can `vault.add()` as much as you want, then when you're done, process all the vectors at once with a `vault.get_vectors()` call - Which internally batches vector embeddings with OpenAI's text-embeddings-ada-002, and comes with auto rate-limiting and concurrent requests for maximum processing speed. 
```python
vault.add(very_large_text)
vault.get_vectors() 
vault.save() 
```
^ these three lines execute fast and can be called as often as you like. For example: you can use `add()`, `get_vectors()`, and `save()` mid conversation to save every message to the vault as soon as they comes in. Small loads are usually finished in less than a second. Large loads depend on total data size. 
>> A test was done adding the full text of 37 books at once. The `get_vectors()` function took 8 minutes and 56 seconds. (For comparison, processing one at a time via openai's embedding function would take roughly two days)

<br>
<br>

# Use The Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/5ae905b0-43d0-4b86-a965-5b447ee8c7de/Vector+Vault+Vault.jpg?content-type=image%2Fjpeg" width="60%" height="60%" />
</p>

You can create a javascript or HTML post to `"https://api.vectorvault.io/get_similar"`, to run front end apps.
Since your Vault lives in the cloud, making a call to it is really easy. You can even do it with a `curl` from command line:
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
>> {"results":[{"data":"NASA Mars Exploration... **shortend for brevity**","metadata":{"created_at":"2023-05-29T19:21:20.846023","item_id":0,"name":"webdump-0","updated_at":"2023-05-29T19:21:20.846028"}}]}
    
This is the same exact call, but in Python:
```python
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
```
>> NASA Mars Exploration... NASA To Host Briefing... Program studies Mars... A Look at a Steep North Polar...

^ this prints each similar item that was retieved. The `get_similar()` function retrieves items from the vault using vector cosine similarity search algorithm to find results. Default returns a list with 4 results. 
`similar_data = vault.get_similar(text_input, n = 10)` returns 10 results instead of 4.

<br>

Print the metadata:
```python
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
    print(result['metadata'])
```
>> NASA Mars Exploration... {"created_at":"2023-05-29T19...} NASA To Host Briefing... {"created_at":"2023-05-29T19...} Program studies Mars... {"created_at":"2023-05-29T19...} A Look at a Steep North Polar... {"created_at":"2023-05-29T19...}

<br>
<br>

# Metadata Made Easy

To add metadata to your vault, just include the meta as a parameter in `add()`. Meta is always a dict, and you can add any fields you want.
```python
meta = {
    'name': 'Lifestyle in LA',
    'country': 'United State',
    'city': 'LA' 
}

vault.add(text, meta)

vault.get_vectors()

vault.save()
```

<br>

To add just the 'name' field to the metadata...
```python
vault.add(text, name='Lifestyle in LA')

vault.get_vectors()

vault.save()
```

<br>

Find the name later:
```python
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['metadata']['name'])
```
>> Lifestyle in LA Lifestyle in LA Lifestyle in LA Lifestyle in LA

<br>

### Add Any Meta Fields & Retrieve later

```python
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


*Notice we are not printing any content, just the metadata:*
```python
# Later
similar_data = vault.get_similar("How will the government control you in the future?") 

for result in similar_data:
    print(result['metadata']['title'])
    print(result['metadata']['author'])
    print(result['metadata']['genre'])
```
>> 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian 1984 George Orwell Dystopian



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
Retrieving items from the vault, is useful when using it supply context to a large language model, like chatgpt for instance, to get a contextualized response. The follow example searches the vault for 4 similar results and then give those to chatgpt as context, asking chatgpt answer the question using the vault data.
```python
question = "Should I use Vector Vault for my next generative ai application"

answer = vault.get_chat(question, get_context=True)  
print(answer)
```
The following line will send chatgpt the question for response and not interact with the vault in any way
```python
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
Response from ChatGPT in string format, unless `return_context=True`, then response will be a dictionary containing response from ChatGPT and the vault data.

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


`get_chat()` function returns the whole message at once. `get_chat_stream` yields each word as it's received. Other than that, they are nearly identical and have the same input parameters.

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

vault = Vault(user='your_user_id', api_key='your_api_key', vault='Customer Service')

with open('customer_service.txt', 'r') as f:
    vault.add(f.read())

vault.get_vectors()

vault.save()
```

<br>

And just like that, in a only a few lines of code we created a customer service vault. Now whenever you want to use it in production, just use the `get_chat()` with `get_context=True`, which will take the customer's question, search the vault to find the most similar questions and answers, then have ChatGPT reply to the customer using that information.

```python
question = 'customer question'

answer = vault.get_chat(question, get_context=True)
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
