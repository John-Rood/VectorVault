![alt text](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a cloud vector database service that was built to make generative ai chat quick and easy. It allows users to vectorize data easily and access them seamlessly from the cloud. It's suitable for both small projects and large. Vector Vault has been designed with a user-friendly code interface to make the process of working with vector data easy and let you focus on what matters, results. Vector Vault ensures secure and isolated data handling and enables you to create and interact vector databases - aka "vaults" - in the cloud, at millisecond response times.

By combining vector similarity search with generative ai chat, new possibilities for conversation and communication emerge. For example, product information can be added to a vault, and when a customer asks a product question, the right product information can be instantly retreived and seamlessly used in conversation by chatgpt for an accurate response. This capability allows for informed conversation and the possibilites range from ai automated customer support, to new ways to get news, to ai code reviews that reference source documentation, to ai domain experts for specific knowledge sets, and much more.

Vector Vault uses a proprietary architecture, called "Inception", allowing you to create any number of vaults, and vaults within a vaults. Each vault is it's own database, and automatically integrates data storage in the cloud. You will need a Vector Vault api key in order to access the cloud vaults. If you don't already have one, you can use the included `register()` function or sign up at [VectorVault.io](https://vectorvault.io)

The `vectorvault` package allows you to interact with your Cloud Vaults using its Python-based API. Each vault is a seperate vector database. `vectorvault` includes operations such as creating a vault, deleting a vault, preparing data to add, getting vector embeddings for prepared data using OpenAI's text-embedding-ada-002, saving the data and embeddings to the cloud, referencing cloud vault data via vector search and retrieval, interacting with OpenAI's ChatGPT model to get responses, managing conversation history, and retrieving contextualized responses with reference vault data as context.

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
`get_similar()` : Retrieves similar texts from the Vault for a given input text - We processes vectors in the cloud
<br>
`get_similar_local()` : Retrieves similar texts from the Vault for a given input text - You process vectors locally
<br>
`get_chat()` : Retrieves a response from ChatGPT, with support for handling conversation history, summarizing responses, and retrieving context-based responses by referencing similar data in the vault

>> `get_vectors()` utilizes openai embeddings api and internally batches vector embeddings with OpenAI's text-embeddings-ada-002 model, and comes with auto rate-limiting and concurrent requests for maximum processing speed


<br>

# Build Your Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/2acebcaa-f5dd-44c9-8bba-c10723bc7064/Vector+Vault+Vault+2000.png" width="60%" height="60%" />
</p>

Install Vector Vault:
```
pip install vector-vault
```
<br>

### Get Your Vector Vault API Key:
```
from vectorvault import register

register(first_name='John', last_name='Smith', email='john@smith.com', password='make_a_password')
```
The api key will be sent to your email.

<br>

# Use Vector Vault:

Set your openai key as an envorionment variable
```
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
```

1. Create a Vault instance - (new vault will be created if name does not exist)
2. Gather some text data we want to store
3. Add the data to the Vault
4. Get vectors embeddings 
5. Save to the cloud vault

```
from vectorvault import Vault

vault = Vault(user='your@email.com', api_key='your_api_key', vault='name_of_vault)

text_data = 'some data'

vault.add(text_data)

vault.get_vectors()

vault.save()
```

<br>
<br>

Now that you have saved some data to the vault, you can add more at anytime, and your vault will automatically handle the adding process. These three lines execute very fast.
```
vault.add(more_text_data)

vault.get_vectors()

vault.save()
```

<br>
<br>

`vault.add()` is very versitile. You can add any length of text, even a full book...and it will be all automatically split and processed.
`vault.get_vectors()` is also extremely flexible, because you can `vault.add()` as much as you want, then when you're done, process all the vectors at once with a `vault.get_vectors()` call - Which internally batches vector embeddings with OpenAI's text-embeddings-ada-002, and comes with auto rate-limiting and concurrent requests for maximum processing speed. 
```
vault.add(insanely_large_text_data)
vault.get_vectors() 
vault.save() 
```
^ these three lines execute fast and can be called as often as you like. For example: you can use `add()`, `get_vectors()`, and `save()` mid conversation to save every message to the vault as soon as they comes in. Small loads are usually finished in less than a second. Large loads depend on total data size. 
>> A test was done adding the full text of 37 books at once. The `get_vectors()` function took 8 minutes and 56 seconds. (For comparison, processing one at a time via openai's embedding function would take roughly two days)

<br>
<br>

# Call Your Vault:
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
```
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
```
>> NASA Mars Exploration... NASA To Host Briefing... Program studies Mars... A Look at a Steep North Polar...

^ this prints each similar item that was retieved. The `get_similar()` function retrieves items from the vault using vector cosine similarity search algorithm to find results. Default returns a list with 4 results. 
`similar_data = vault.get_similar(text_input, n = 10)` returns 10 results instead of 4.

<br>

Print the metadata:
```
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
    print(result['metadata'])
```
>> NASA Mars Exploration... {"created_at":"2023-05-29T19...} NASA To Host Briefing... {"created_at":"2023-05-29T19...} Program studies Mars... {"created_at":"2023-05-29T19...} A Look at a Steep North Polar... {"created_at":"2023-05-29T19...}

<br>
<br>

### Use `get_chat()` with `get_context=True` to get response from chatgpt referencing vault data
Retrieving items from the vault, is useful when using it supply context to a large language model, like chatgpt for instance, to get a contextualized response. The follow example searches the vault for 4 similar results and then give those to chatgpt as context, asking chatgpt answer the question using the vault data.
```
question = "Should I use Vector Vault for my next generative ai application"

answer = vault.get_chat(question, get_context=True)  
print(answer)
```
The following line will send chatgpt the question for response and not interact with the vault in any way
```
answer = vault.get_chat(question) 
```


<br>
<br>

# ChatGPT
## With `get_chat()` you can use ChatGPT standalone or with Vault data integrated

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
<br>

Response from ChatGPT in string format, unless `return_context=True` is passed, then response will be a dictionary containing the results - response from ChatGPT, and the vault data.
```
# print response:
print(vault_response['response'])

# print context:
for item in vault_response['context']:
    print("\n\n", f"item {item['metadata']['name']}")
    print(item['data'])
```

<br>
<br>

# Summarize Anything:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/e1ff4ca3-e18b-4c8f-b3c9-ff6ddcc907a1/wisdomandwealth_a_summary_being_created._A_bunch_of_texts_are_f_df58744a-13cb-46fd-b39d-3f090349bbb7.png" width="60%" height="60%" />
</p>

You can summarize any text, no matter how large - even an entire book all at once. Long texts are split into the largest possible chunk sizes and a summary is generated for each chunk. When all summaries are finished, they are concatenated and returned as one.
```
summary = vault.get_chat(text, summary=True)
```
<br>

want to make a summary of a certain length?...
```
summary = vault.get_chat(text, summary=True)

while len(summary) > 1000:
    summary = vault.get_chat(summary, summary=True)
```
^ in the above example, we make a summary, then we enter while loop that continues until the summary recieved back is a certain lenght. You could use this to summarize a 1000 page book to less than 1000 characters of text. 

<br>
<br>
<br>

# Streaming:
Use the built in streaming functionality to get interactive chat streaming. Here's an [app](https://philbrosophy.web.app) we built to showcase what you can do with Vector Vault:
<br>

![Alt text](https://github.com/John-Rood/VectorVault/blob/main/examples/Streaming%20Demo%20Offish.gif)

## get_chat_stream():

Example Usage: `vault.print_stream(vault.get_chat_stream(text))`
Always use this `get_chat_stream()` wrapped by either `print_stream()` or `cloud_stream()`.
`cloud_stream()` is for cloud functions, like a flask app serving a front end elsewhere.
`print_stream()` is for local console printing.

Example Signle Usage: 
`response = vault.print_stream(vault.get_chat_stream(text))`

Example Chat: 
`response = vault.print_stream(vault.get_chat_stream(text, chat_history))`

Example Summary: 
`summary = vault.print_stream(vault.get_chat_stream(text, summary=True))`

Example Context-Based Response:
`response = vault.print_stream(vault.get_chat_stream(text, get_context = True))`

Example Context-Based Response w/ Chat History:
`response = vault.print_stream(vault.get_chat_stream(text, chat_history, get_context = True))`

Example Context-Response with Context Samples Returned:
`vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True))`

Example Context-Response with SPECIFIC META TAGS for Context Samples Returned:
`vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author']))`

Example Context-Response with SPECIFIC META TAGS for Context Samples Returned & Specific Meta Prefixes and Suffixes:
`vault_response = vault.print_stream(vault.get_chat_stream(text, get_context = True, return_context = True, include_context_meta=True, metatag=['title', 'author'], metatag_prefixes=['\n\n Title: ', '\nAuthor: '], metatag_suffixes=['', '\n']))`

Response is a always a stream
`vault.get_chat_stream` will start a chat stream. The input parameters are mostly like the regular get_chat functionality, and the capabilites are all the same. The only difference is that the get_chat function returns the whole reply message at once. The get_chat_stream `yield`s each word as it it received. This means that using `get_chat_stream()` is very different than using `get_chat()`. Here's an example that prints the same message:

```
## get_chat()
print(vault.get_chat(text, history)

## get_chat_stream()
for word in vault.get_chat_stream(text, history):
        print(word)
```
This will take each word yielded and print it as it comes in. However, that will not look good, so it's best to use the built in print function `print_stream`. 
```
vault.print_stream(vault.get_chat_stream(text, history))
```
<br>

Because streaming is a key functionality for end user applications, we also have a `cloud_stream` function to make cloud streaming to your front end app easy. In a flask app, your return would look like: `return Response(vault.cloud_stream(vault.get_chat_stream(text, history, get_context=True)), mimetype='text/event-stream')`
This makes going live with highly functional cloud apps really easy. Now you can build impressive applications in record time! If have any questions, message in [Discord](https://discord.gg/AkMsP9Uq).


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
```
from vectorvault import Vault

os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

vault = Vault(user='your_user_id', api_key='your_api_key', vault='Customer Service')

with open('customer_service.txt', 'r') as f:
    vault.add(f.read())

vault.get_vectors()

vault.save()
```

<br>

And just like that, in a only a few lines of code we created a customer service vault. Now whenever you want to use it in production, just connect to that vault, and use the `get_chat()` with `get_context=True`. When you call `get_chat(text, get_context=True)` it will take the customer's question, search the vault to find the most similar questions and answers, then have ChatGPT reply to the customer using that information.

```
question = 'customer question'

answer = vault.get_chat(question, get_context=True)
```
<br>

That's all it takes to create an AI customer service chatbot that responds as well as any support rep!


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


## FAQ
<br>

### What is the latency on large datasets?
37 full length book texts with vectors make up ~250MB of storage with around 10,000 - 15,000 items of 1000+ characters for each item. This example of 37 books is considered small. Personal plans come with 1GB of storage, so this doesn't even come close to plan limit. Calling similar items from this vault is one second response time - with vectors retreived, vectors searched, then similar items returned. Running locally, you can expect ~0.7 seconds. Just referencing the cloud, you can expect ~1 second. This example is about the same amount of data as the entire customer support history for any given company. So if you build a typical customer service chatbot, your vault size will be considered small. You can try our app at the bottom of our [website](https://vectorvault.io) to see the latency for yourself. Keep in mind that if you do so, you will be seeing the latency from chatgpt on top of the vault time, but it's still so fast that its not noticible during a conversation. If you had 10 times that much data, api latency may be around 2 seconds max, so still fast enough for realtime conversations. Or you can just get an Enterprise Cloud Plan from us and get it even faster.

<br>

### How should I segment my data?
Vaults within vaults is the optimal structure for segmenting data. If a vault grows too large, just make multiple child vaults within the current vault directory, and store the data there. If your 'Science' vault grows too large, split it into multiple child vaults, like 'Science/Chemistry', etc - this accesses a "Chemistry" vault within the Science vault. Now you can fine grain datasets, where every child vault contains more specific subject information than the parent vault. This segmenting structure allows you to focus data on large data sets.

<br>

### What if I'm a large company with very large data
If you need to store more than 1 gig of data in single vaults for any reason, let us know and we can set you up with Enterprise Cloud Plan. In our Enterprise plan, we create a persistent storage pod with as much memory as you need. It is always active and scalable to terabytes. With an Enterprise plan, a billion vectors search will respond in one second. For reference, the full text of 3.7 million books would be ~1.1 billion vectors, and take up ~8 terabytes of storage. If this is what you're looking for, just reach out to us by email at support at vectorvault.io.

