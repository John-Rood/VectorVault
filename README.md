VectorVault is designed to simplify the process of working with vector databases. It allows users to vectorize datasets efficiently, and access them seamlessly from the cloud. It's scalable and suitable for both small and large scale projects. VectorVault has been designed with a user-friendly interface to make the process of working with vector databases easy and let you focus on what matters. It simplifies complex workflows, ensures secure and isolated data handling, and enables users to create and interact vector databases - aka "vaults" 

VectorVault was built with the goal of making generative ai work flows simple and easy. By combining vector similarity search with generative ai chat, new possibilities for conversation and communication emerge. For example, product information can be added to a vault, and when a customer asks a product question, the right product information can be instantly retreived and seamlessly used in conversation by chatgpt for an accurate response. This capability allows for informed conversation and the possibilites range from ai automated customer support, to new ways to get news, to ai code reviews that reference source documentation, to ai domain experts for specific knowledges, and much more.

VectorVault uses a proprietary Inception Architecture, allowing you to create any number of vaults, and vaults within a vaults. Each vault is it's own database, and automatically integrates data storage in the cloud. You will need a VectorVault account in order to get your user id and api key for cloud access. If you don't already have one, you can sign up free at [VectorVault.io](https://vectorvault.io)

This python library allows you to interact with VectorVault using its Python-based API. It includes operations such as creating a vault, deleting the vault, adding data to the vault, getting vector embeddings for the data, saving data to the vault, interacting with OpenAI's ChatGPT model to get responses, and managing conversation history for more contextualized responses.

<br>

# Interact with your Vault:

`add` : Add item to the Vault, with automatic text splitting and processing for long texts. Main way to add to vault
<br>
`add_item` : Add item to the Vault
<br>
`add_item_with_vector` : Add item to the Vault with vector provided - only accepts vectors of 1536 dimensions
<br>
`save` : Saves the vectors to the Vault and uploads any metadata
<br>
`delete` : Deletes the current Vault
<br>
`get_vaults` : Retrieves a list of vaults in the current vault 
<br>
`get_similar` : Retrieves similar vectors for a given input text
<br>
`get_vectors` : Retrieves the vectors for all items in the Vault
<br>
`get_chat` : Retrieves a response from OpenAI's ChatGPT for a given input text, with support for handling conversation history, summarizing responses, and retrieving context-based responses by accessing similar references in the vault


<br>
<br>

## Basic usage:
Install VectorVault:
```
pip install vector-vault
```
Use VectorVault:
```
from vectorvault import Vault

# Create an instance of the Vault class - a new vault will be created if name does not exist
vault = Vault(user='your_user_id', api_key='your_api_key', vault='name_of_your_vault)

# Some text data we want to store
text_data = 'some data'

# Add the data to the Vault
vault.add(text_data)

# add your openai key to environment variable
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

# Get vecctor embeddings for text_data 
# Internally calls openai with automatic rate limiting built in. Large inputs are batched, and concurrently processed for fastest possible embed time.
vault.get_vectors()

# Save the vectors and data to the Vault 
vault.save()
```

<br>
<br>

Now that you have saved some data to the vault, you can add more at anytime, and your vault will automatically handle the adding process. These three lines execute very fast.
```
# Add more data to the Vault
vault.add(more_text_data)

# Get embeddings for it - requires an openai api key set as an environvment variable
vault.get_vectors()

# Save to the Vault
vault.save()
```

<br>
<br>

vault.add() is cool. You can add any length of text, even a full book...and it will be all automatically split and processed.
vault.get_vectors() is also cool, because you can vault.add() as much as you want, then when you're done, process all the vectors at once with a vault.get_vectors() - Internally batches vector embeddings with OpenAI's text embeddings ada 002, and comes with auto rate-limiting and concurrent requests for maximum processing speed


```
vault.add(insanely_large_text_data)
vault.get_vectors() 
vault.save() 
```
^ these three lines execute fast and can be called as often as you like. For example: `add`, `get_vectors`, and `save` can be used mid conversation to add every message to the vault one at a time as they comes in.


<br>
<br>

## Use the vault later:
```
similar_data = vault.get_similar(text_input) # returns a list with 4 results
similar_data = vault.get_similar(text_input, n = 10) # returns 10 results

# Print each similar item 
for result in similar_data:
    print(result['data'])
```


<br>

## Use the get_chat() function to get a response from chatgpt
The following searches the vault for 4 similar results and then give those to chatgpt as context, asking chatgpt answer the question using the context
```
question = "This text is going to be used find contextually similar references in the vault"

answer = vault.get_chat(question, get_context=True)  
print(answer)

# The following line will send chatgpt the question for response and not interact with the vault in any way
answer = vault.get_chat(question) 
```


<br>
<br>

# Change Vault
In this example science vault, we will print a list of vaults in the current vault directory
```
science_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science')

print(science_vault.get_vaults())
```
>> ['biology', 'physics', 'chemistry']



## Access vaults within vaults

- biology vault within science vault
```
biology_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/biology')
```



- chemistry vault within science vault
```
chemistry_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/chemistry')

print(chemistry_vault.get_vaults())
```
>> ['reactions', 'formulas', 'lab notes']



- lab notes vault within chemistry vault
```
lab_notes_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/chemistry/lab notes')
```

<br>
<br>

# get_chat()
Chat get response from OpenAI's ChatGPT. 
Rate limiting, auto retries, and chat histroy slicing built-in so you can chat with ease. 
Enter your text, add optional chat history, and optionally choose a summary response (default: summmary = False)

- Example Signle Usage: 
`response = vault.get_chat(text)`

- Example Chat: 
`response = vault.get_chat(text, chat_history)`

- Example Summary: 
`summary = vault.get_chat(text, summary=True)`

- Example Context-Based Response:
`response = vault.get_chat(text, get_context = True)`

- Example Context-Based Response w/ Chat History:
`response = vault.get_chat(text, chat_history, get_context = True)`

- Example Context-Response with Context Samples Returned:
`vault_response = vault.get_chat(text, get_context = True, return_context = True)`
<br>

Response is a string, unless `return_context = True` is passed, then response will be a dictionary containing the results from the vault as well as the response:
```
# print response:
print(vault_response['response'])` 

# print context:
for item in vault_response['context']['results']:
    print("\n\n", f"item {item['metadata']['item_id']}")
    print(item['data'])
```

# Summarize Anything:
You can summarize any text, no matter how large - even an entire book all at once. Long texts are split into the largest possible chunk sizes and a summary is generated for each chunk. When all summaries are finished, they are concatenated and returned as one.
```
summary = vault.get_chat(text, summary=True)
```

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
for item in vault_response['context']['results']:
    print("\n\n", f"item {item['metadata']['item_id']}")
    print(item['data'])

```

>> Question: 
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
>> Question: 
How do I use it? 
 
>>Answer: 
You can use it by...



<br>
<br>
<br>

# Build an AI Cusomter Service Chat Bot
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

And just like that, in a only a few lines of code we created a customer service vault based on all customer support messages. Now whenever you want to use it in the wild, just connect to that vault, and use the `get_chat()` function with `get_context=True`. The call `get_chat(text, get_context=True)` will take the customer's question, search the vault, find the most similar questions and answers, then have ChatGPT reply to the customer using that information.

```
question = 'customer question text string'

answer = vault.get_chat(question, get_context=True)
```

That's all it takes to create an ai customer service chatbot that responds to your customers as well as any support rep!


<br>
<br>

If have any questions, leave a comment. Open the "examples" folder and try out the Google Colab tutorials we have.

Happy coding.

