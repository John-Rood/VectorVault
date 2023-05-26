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
`get_similar()` : Retrieves similar texts from the Vault for a given input text 
<br>
`get_chat()` : Retrieves a response from ChatGPT, with support for handling conversation history, summarizing responses, and retrieving context-based responses by referencing similar data in the vault

>> `get_vectors()` utilizes openai embeddings api and internally batches vector embeddings with OpenAI's text-embeddings-ada-002 model, and comes with auto rate-limiting and concurrent requests for maximum processing speed


<br>

# Interact with your Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/3a6c60a3-79ac-467c-b640-c434499ca76d/Vector+Vault+Vault+2000.jpg" width="60%" height="60%" />
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

# Reference Your Vault:
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/5ae905b0-43d0-4b86-a965-5b447ee8c7de/Vector+Vault+Vault.jpg?content-type=image%2Fjpeg" width="60%" height="60%" />
</p>

After you've added some data and want to reference it later, you can call it like this:
```
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['data'])
```
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

<br>

## Metadata
To add meta data to your vault, just include the meta as a parameter in `add()`. Meta is always a dict, and you can add any fields you want. (If you don't add a 'name' field, a generic one will automatically be generated, so there is always a name field in the metadata)
```
metadata = {
    'name': 'Lifestyle in LA',
    'country': 'United State',
    'city': 'LA' 
}

vault.add(text_data, metadata)

vault.get_vectors()

vault.save()
```

<br>

To add just the 'name' field to the metadata, call the `name` param in `add()` like this:
```
vault.add(more_text_data, name='Lifestyle in LA')

vault.get_vectors()

vault.save()
```

<br>

To find the name later:
```
similar_data = vault.get_similar("Your text input") 

for result in similar_data:
    print(result['metadata']['name'])
```
<br>

### Add Any Meta Fields & Retrieve later
Here we open the popular book by George Orwell, '1984' from a .txt file. We read the file and save all the book's text to a variable called 'text'. Then we create a dictionary containing all the information about the book. Then we save all that to the vault. When we call later, we can reference any of the meta data. The vault reference will return sample text from the book, and if the vault has many books in it, you may want to know what exactly you're referencing. The metadata is how you will know.
```
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

To find the metadata later:
```
similar_data = vault.get_similar("How will the government control you in the future?") 

for result in similar_data:
    print(result['metadata']['title'])
    print(result['metadata']['author'])
    print(result['metadata']['genre'])
    # etc...
```
^ list is always returned. So you can break it down like this too...

```
similar_data = vault.get_similar("How will the government control you in the future?") 
print(result[0]['metadata']['title'])
```

<br>
<br>

### Use `get_chat()` with `get_context=True` to get response from chatgpt referencing vault data
Retrieving items from the vault, or texts, is useful when using it supply context to a large language model, chatgpt for instance, to get a contextualized response. The follow example searches the vault for 4 similar results and then give those to chatgpt as context, asking chatgpt answer the question using the vault data.
```
question = "This text will be used find contextually similar references in the vault"

answer = vault.get_chat(question, get_context=True)  
print(answer)
```
The following line will send chatgpt the question for response and not interact with the vault in any way
```
answer = vault.get_chat(question) 
```


<br>
<br>

# Change Vaults

In this example science vault, we will print a list of vaults in the current vault directory
```
science_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science')

print(science_vault.get_vaults())
```
>> Output: ['biology', 'physics', 'chemistry']


## Access vaults within vaults

- biology vault within science vault
```
biology_vault = Vault(user='your@email.com', api_key='your_api_key', vault='science/biology')
```

- chemistry vault within science vault
```
chemistry_vault = Vault(user='your@email.com', api_key='your_api_key', vault='science/chemistry')

print(chemistry_vault.get_vaults())
```
>> Output: ['reactions', 'formulas', 'lab notes']


- lab notes vault within chemistry vault
```
lab_notes_vault = Vault(user='your@email.com', api_key='your_api_key', vault='science/chemistry/lab notes')
```

<br>
<br>

# ChatGPT
## With `get_chat()` you can use chatgpt standalone or with vault data integrated

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

Response is a string, unless `return_context=True` is passed, then response will be a dictionary containing the results from the vault as well as the response:
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

## Real world usage:
```
user_input = input("What's your question?")

# Get response from Language model
vault_response = vault.get_chat(user_input, get_context=True, return_context=True)

answer = vault_response['response']
print("Question:", user_input, "\n\nAnswer:", answer)

# show the context used to generate the answer
for item in vault_response['context']:
    print("\n\n", f"item {item['metadata']['item_id']}")
    print(item['data'])

```

>> Output: Question: 
What is a token broker? 
 
>>Answer: 
A token broker is a service that generates downscoped access tokens for token
consumers to access or modify specific resources...
>>

>> item 33
Various workloads (token consumers) in the same network will send authenticated
requests to that broker for downscoped tokens to...
>>
 >>item 4
Another reason to use downscoped credentials is to ensure tokens in flight...
>>


>>
>> item 37
The following is an...
>>


<br>

```
user_input2 = input("What's your next question?")

history = user_input + answer

# Get response from Language model
vault_response = vault.get_chat(user_input2, history=history, get_context=True)

print("Question:", user_input2, "\n\nAnswer:", vault_response2)
```
>> Output: Question: 
How do I use it? 
 
>>Answer: 
You can use it by...



<br>
<br>
<br>

# Build an AI Cusomter Service Chat Bot
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/dceb5c7d-6ec6-4eda-82f2-b8848c7b519d/ai_chatbot_having_a_conversation.png" width="60%" height="60%" />
</p>
<br>

In the following code, we will add all the customer conversations that a support team has ever had to a vault. (Assumes you have all the past conversations downloaded to a single text file). Then we we will take cusotmer requests, search the database for similar questions and answers. After that, we will instruct ChatGPT to use the previous answers the support team has given to answer this new question. (NOTE: This will also work based on a customer FAQ, or customer support response templates).

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

And just like that, in a only a few lines of code we created a customer service vault based on all customer support messages. Now whenever you want to use it in the wild, just connect to that vault, and use the `get_chat()` with `get_context=True`. When you call `get_chat(text, get_context=True)` it will take the customer's question, search the vault to find the most similar questions and answers, then have ChatGPT reply to the customer using that information.

```
question = 'customer question text string'

answer = vault.get_chat(question, get_context=True)
```

That's all it takes to create an ai customer service chatbot that responds to your customers as well as any support rep!


<br>
<br>

If have any questions, leave a message/comment on github repo. 

<br>

Open the "examples" folder and try out the Google Colab tutorials we have.

Happy coding!
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/7d1a596b-7560-446b-aa69-1827819d198b/Looking+out+with+hope+vector+vault.png" width="60%" height="60%" />
</p>

