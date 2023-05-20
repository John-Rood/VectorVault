Vector Vault is designed to simplify the process of working with vector databases. It allows users to manage vector databases efficiently, integrated and accessed seamlessly from the cloud. It's scalable, suitable for both small and large scale databases, and designed with a user-friendly interface. Furthermore, it simplifies complex workflows, ensures secure and isolated data handling, and enables users to create and interact vector databases - aka "vaults" - simply and easily.

Vector Vault was built with the goal of making complex work flows, that utilize vector databases for informed generative ai, simple and easy. By combining similarity vector search with generative ai chat, new possibilities for conversation and communication emerge. Product information can be added to a vault, and then when customers ask a product question, the right information can be instantly retreived and seamlessly used in conversation by chatgpt for an accurate response. This allows for informed conversation and the possibilites range from ai automated customer support, to new ways to get news and entertainment, to ai code reviews that reference documentation, and much more.

Vector Vault uses a proprietary Inception Architecture, allowing you to create any number of vaults, and vaults within a vaults. Each vault is it's own database, and automatically integrates data storage in the cloud. You will need a Vector Vault account in order to get your user id and api key for cloud access. If you don't already have one, you can sign up free at https://vectorvault.io

This python library allows you to interact with Vector Vault using its Python-based API. It includes operations such as creating a vault, deleting the vault, adding data to the vault, getting vector embeddings for the data, saving data to the vault, interacting with OpenAI's ChatGPT model to get responses, and managing conversation history for more contextualized responses.

There are several methods to interact with your Vault, such as:

`add` : Add item to the Vault, with automatic text splitting and processing for long texts. Main function to add to the vault.

`add_item` : Add item to the Vault

`add_item_with_vector` : Add item to the Vault with vector externally provided, although it only accepts vectors of 1536 dimensions

`save` : Saves the vectors to the Vault and uploads any metadata.

`delete` : Deletes the current Vault.

`get_vaults` : Retrieves a list of vaults in the current vault 

`get_similar` : Retrieves similar vectors for a given input text.

`get_vectors` : Retrieves the vectors for all items in the Vault.

`get_chat` : Retrieves a response from OpenAI's ChatGPT for a given input text, with support for handling conversation history, summarizing responses, and retrieving context-based responses by accessing similar references in the vault.



These methods collectively allow you to create and access your vector databases aka "vaults", which are securely saved in cloud storage.


Basic usage:
```
from vector_vault import Vault

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



Now that you have saved some data to the vault, you can add more at anytime, and your vault will automatically handle the adding process. These three lines execute very fast.
```
# Add more data to the Vault
vault.add(more_text_data)

# Get embeddings for it - requires an openai api key set as an environvment variable
vault.get_vectors()

# Save to the Vault
vault.save()
```



vault.add() is cool. You can add any length of text, even a full book...and it will be all automatically split, processed, and added.
```
vault.add(insanely_large_text_data)
vault.get_vectors() 
vault.save() 
```





## When you want to use the vault later:
```
similar_data = vault.get_similar(text_input) # returns 4 results
similar_data = vault.get_similar(text_input, n = 10) # returns 10 results

# Print each similar item 
for result in similar_data:
    print(result['data'])
```


## Use the get_chat() function to get a response from chatgpt
The following searches the vault for 4 similar results and then give those to chatgpt as context, asking chatgpt answer the question using the context
```
user_input = "Is this question going to be used find contextually similar references in the vault?"

answer = vault.get_chat(user_input, get_context=True)  
print(answer)

# The following line will just send chatgpt the user_input and not interact with the vault in any way
answer = vault.get_chat(user_input) 
```

## Change vault directory
### Example:
```
science_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science')

# Print a list of vaults in the current vault directory
print(science_vault.get_vaults())
```
>> ['biology', 'physics', 'chemistry']

## Access vaults within vaults
```
biology_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/biology')
```

chemistry vault within science vault
```
chemistry_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/chemistry')
print(chemistry_vault.get_vaults())
```
>> ['reactions', 'formulas', 'lab notes']

lab notes vault within chemistry vault
```
lab_notes_vault = Vault(user='your_user_id', api_key='your_api_key', vault='science/chemistry/lab notes')
```