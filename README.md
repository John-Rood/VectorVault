# Vector Vault

![Vector Vault Header](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a cutting-edge, cloud-native and RAG-native vector database solution that revolutionizes AI integration in applications. Our platform seamlessly combines vector databases, similarity search, and AI model interactions into a single, easy-to-use service.

## Key Features

- **RAG-Native Architecture**: Perform Retrieval-Augmented Generation in one line of code.
- **Unparalleled Simplicity**: Implement sophisticated AI features with minimal code.
- **Full-Stack Integration**: Use our Python package for backend operations and our JavaScript package for easy front-end integration.
- **Cloud-Engine**: Our service handles vector search, retrieval, and AI model interactions, simplifying your architecture.
- **One-Line Operations**: Save to the cloud vector database and generate RAG responses in one line of code.
- **Developer-Centric**: Focus on your application logic rather than complex AI and front-end integrations.
- **Unlimited Isolated Databases**: Create and access an infinite number of vector databases, ideal for multi-tenant applications.

## Quick Start

Install Vector Vault:
```bash
pip install vector-vault
```

Basic usage:
```python
from vectorvault import Vault

vault = Vault(user='YOUR_EMAIL',
              api_key='YOUR_API_KEY', 
              openai_key='YOUR_OPENAI_KEY',
              vault='NAME_OF_VAULT')

# Add data to your vault
vault.add('some text')
vault.get_vectors()
vault.save()

# Get AI-powered RAG responses
rag_response = vault.get_chat("Your question here", get_context=True)
print(rag_response)
```

# DeepSeek Update
```python
# DeepSeek's V3 chat model:
response = vault.get_chat("Your question here", get_context=True, model='deepseek-chat')
print(response)

# DeepSeek's R1 reasoning model :
response = vault.get_chat("Your question here", get_context=True, model='deepseek-reasoner')
print(response)
```
Learn more about DeepSeek V3 [here](https://api-docs.deepseek.com/news/news1226).
<br>
Learn more about DeepSeek R1 [here](https://api-docs.deepseek.com/news/news250120).



## Key Concepts

- **Vaults**: Isolated serverless Vector databases. No limits, inifitely scalable.
- **RAG-Native**: Vector Similarity Search - Retrieval Augmented Generatioy - fully customizable with params
- **Cloud Engine**: We process operations and AI references in the Vector Vault cloud, making it easy for you to integrate to the front end quickly and securely

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

- **Simplicity**: Easier to use than traditional vector databases and AI integrations.
- **RAG Optimization**: Built from the ground up for Retrieval-Augmented Generation workflows.
- **Customization**: Add specific knowledge to your Vault and tailor AI responses to your needs.
- **Scalability**: Fully serverless platform offering unparalleled scalability.
- **Time and Resource Saving**: Dramatically reduce development time for AI feature integration.

## Getting Started

1. Sign up for a 30-day free trial at [VectorVault.io](https://vectorvault.io) to get your API key.
2. Install the `vectorvault` package: `pip install vector-vault`
3. Explore our [examples folder](https://github.com/John-Rood/VectorVault/tree/main/examples) for tutorials and practical applications.

## Learn More

- Full API Documentation: [Link to API docs](https://github.com/John-Rood/VectorVault/tree/main/fulldocs.md)
- Interactive Dashboard: [app.vectorvault.io](https://app.vectorvault.io)
- Join our [Discord community](https://discord.gg/AkMsP9Uq) for support and discussions.

Start building with Vector Vault today and experience the future of RAG-native, cloud-native vector databases!

## A Note from the Author
I built this platform to save time and help go from idea to reality faster. You can deploy in an instant and get to the front end seamlessly. It's a vertically integrated Agentic management platform. What that means is that the vector database, cloud services, and agentic flow builder was all built in-house and fully integrated from the start. Everything works together really well, super fast, and always reliably. Vector Vault is hosting agentic interactions every day for dozens of enterprises around the globe. I wanted to make the platform free initially, but I needed to revert to pay model due to malicious attacks and API abuse. The cloud storage and services cost money and a pay wall is the easiest way to keep out bad actors, but I made it 30 days free so you can check it out and see if you like it. You can click on billing and cancel at any time, so no worries. Sign up and see how much faster and easier it is to develop RAG Apps and AI Agents with Vector Vault.