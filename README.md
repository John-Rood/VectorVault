# Vector Vault

![Vector Vault Header](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a cutting-edge, cloud-native and RAG-native vector database solution that revolutionizes AI integration in applications. Our platform seamlessly combines vector databases, similarity search, and AI model interactions into a single, easy-to-use service. See our [full docs](https://github.com/John-Rood/VectorVault/tree/main/fulldocs.md) or our [JS repo](https://github.com/John-Rood/VectorVault-js) for more information.

## Key Features


- **Simple**: Implement sophisticated AI features with minimal code.
- **Full-Stack**: Use our Python package for backend operations and our JavaScript package for easy front-end integration.
- **RAG-Native**: Perform Retrieval-Augmented Generation in one line of code.
- **Cloud Engine**: Our service handles vector search, retrieval, and AI model interactions, simplifying your architecture.
- **Platform-Agnostic**: Change models without changing your code (OpenAI, Anthropic, Groq, etc.)
- **Unlimited Isolated Databases**: Create and access an infinite number of vector databases, ideal for multi-tenant applications.


## Quick Start

#### Install:
```bash
pip install vector-vault
```

#### Basic Usage:

```python
from vectorvault import Vault

# Initialize Vault
vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_VECTOR_VAULT_API_KEY', 
    openai_key='YOUR_OPENAI_API_KEY',
    vault='MY_NEW_VAULT',
    verbose=True
)

# Build your vault
vault.add('some text') # automatic chunk sizing
vault.get_vectors() # generate vectors for the all data 
vault.save() # save data and vectors to the cloud

# Get AI-powered RAG responses
rag_response = vault.get_chat("What is this vault about?", get_context=True)
print(rag_response)
```
<br>
---------------------------------------------

## Platform Agnostic:
Vector Vault supports multiple AI model platforms - OpenAI, Anthropic, Groq, Grok, and more - all under the same interface. Simply pass in the appropriate API keys upon initialization:
```python
vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_VECTOR_VAULT_API_KEY', 
    openai_key='YOUR_OPENAI_API_KEY',      
    anthropic_key='YOUR_ANTHROPIC_API_KEY', # optional 
    groq_key='YOUR_GROQ_API_KEY',           # optional 
    grok_key='YOUR_GROK_API_KEY',           # optional 
    vault='MY_NEW_VAULT',
    verbose=True
)
```

No matter which provider you choose, downstream methods like get_chat(...) remain the same. You can seamlessly switch providers later without rewriting your code.

<br>
---------------------------------------------

## Adding Personality & Custom Prompts
Vector Vault allows you to define a global “personality” for your AI responses, as well as custom prompts for both context-based and non-context-based queries. This is extremely helpful for brand consistency, specialized tones, or role-playing scenario

#### Setting a Personality
```python
# Define your brand’s or chatbot’s personality
personality_text = """You are an enthusiastic and helpful assistant 
that always uses uplifting language and friendly emojis 😄."""
vault.save_personality_message(personality_text)
```

Once saved, this personality is automatically used in all future responses from this vault.


<br>
---------------------------------------------

## Custom Prompts
You can also set custom prompts that will wrap your user’s message before sending to the model.

```python
# For RAG responses, vector similar data is injected into `context`, and `content` is the user's message 
context_prompt = """You have access to the following context: {context}
Answer using a formal tone:
{content}"""
vault.save_custom_prompt(context_prompt, context=True)
```
Now, whenever you do:

```python
response = vault.get_chat("What's new in the world of data science?", get_context=True)
```
Vector Vault automatically uses your context_prompt before sending to the LLM. By editing this custom_prompt, you can ensure your RAG responses come out perfect every time.

<br>
---------------------------------------------


## Key Concepts

- **Vaults**: Serverless vector databases. Create as many as you need.
- **RAG-Native**: Add data, ask questions, retrieve relevant context from your Vault, and generate AI responses in one step.
- **Cloud Engine**: Our backend handles heavy lifting and integrates seamlessly with multiple AI providers.
- **Personality & Custom Prompts**: Easily store, retrieve, and modify custom roles/tones/prompts for brand consistency.
- **Provider Agnosticism**: Switch from OpenAI to Anthropic or any other platform by changing a single parameter. The rest of your code stays the same.

## Advanced Features

- **Metadata Management**: Easily add and retrieve metadata for your vector entries.
- **Streaming Responses**: Use `get_chat_stream()` for interactive chat experiences.
- **Custom Prompts and Personalities**: Tailor AI responses to your specific needs.

## Use Cases

- AI-powered customer service chatbots
- Semantic search in large document collections
- Personalized content recommendations
- Intelligent chatbots with access to vast knowledge bases
- Multi-tenant systems needing isolated vector databases

## Why Vector Vault?

- **Simplicity**: More straightforward than rolling your own vector database or hooking up multiple AI integrations.
- **RAG Optimization**: Built from the ground up for Retrieval-Augmented Generation workflows.
- **Customization**: Override prompts, personalities, or entire models with minimal code.
- **Scalability**: Serverless approach means no scaling overhead. Build prototypes or enterprise apps all the same.
- **Time and Resource Saving**: Drastically reduce your AI development lifecycle.

## Getting Started

1. Sign up for a 30-day free trial at [VectorVault.io](https://vectorvault.io) to get your API key.
2. Install the `vectorvault` package: `pip install vector-vault`
3. Explore our [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) for tutorials and practical applications.

## Learn More

- Full API Documentation: [Link to API docs](https://github.com/John-Rood/VectorVault/tree/main/fulldocs.md)
- Interactive Dashboard: [app.vectorvault.io](https://app.vectorvault.io)
- Join our [Discord community](https://discord.gg/AkMsP9Uq) for support and discussions.

Start building with Vector Vault today and experience the future of RAG-native, cloud-native vector databases!
