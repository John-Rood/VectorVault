# get_chat() Function Documentation

The `get_chat()` function provides a flexible interface for interacting with AI models, supporting features like conversation history, context-based responses through vector similarity search, and custom prompting capabilities.

## Function Signature

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
    timeout: int = 300           # API timeout in seconds
)
```

## Key Features

### Basic Response
Get a direct response from the AI model without additional context or features.
```python
response = vault.get_chat(text)
```

### Conversation History
Include previous conversation context for more coherent responses.
```python
response = vault.get_chat(text, chat_history)
```

### Text Summarization
Generate a concise summary of the input text.
```python
summary = vault.get_chat(text, summary=True)
```

### Context-Enhanced Responses (RAG)
Retrieve and utilize relevant context through vector similarity search.
```python
# Basic context-based response
response = vault.get_chat(text, get_context=True)

# Context-based response with conversation history
response = vault.get_chat(text, chat_history, get_context=True)

# Include context items in response
response = vault.get_chat(text, get_context=True, return_context=True)
```

## Advanced Features

### Custom Prompting
Custom prompts can be used to modify the AI's response style or behavior. Important notes:
- Only use custom prompts when `get_context=True`
- Custom prompts must include `{context}` and `{question}` placeholders
- For general prompt customization, use the personality message feature instead

```python
my_prompt = """Based on this context: {context}
Please answer this question: {question}
Respond in the style of a financial advisor."""

response = vault.get_chat(text, chat_history, get_context=True, custom_prompt=my_prompt)
```

### Smart History Search
Enable AI-powered search query generation based on conversation history:
```python
response = vault.get_chat(text, chat_history, get_context=True, smart_history_search=True)
```

### Model Selection and Parameters
Fine-tune the response generation:
```python
response = vault.get_chat(
    text,
    model="gpt-4",
    temperature=0.7,
    timeout=60
)
```

## Getting Started

1. Initialize the Vault connection:
```python
vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_API_KEY',
    openai_key='YOUR_OPENAI_KEY',
    vault='vectorvault'
)
```

2. Make your first query:
```python
question = "Why should I use Vector Vault for my next generative ai application?"
answer = vault.get_chat(question, get_context=True)
print(answer)
```

## Error Handling and Rate Limiting

The function includes built-in features for:
- Automatic retry logic
- Rate limiting
- Token management for large inputs
- Error handling with backoff strategy

## Best Practices

1. Use the personality message feature for general prompt customization instead of custom prompts
2. Enable `get_context=True` when domain-specific knowledge is required
3. Consider using `smart_history_search=True` for complex conversations where context from history is important
4. Set appropriate `temperature` values based on your needs:
   - 0: Deterministic, focused responses
   - 0.5: Balanced creativity
   - 1: Maximum creativity
5. Adjust `timeout` based on expected response times and your application's needs