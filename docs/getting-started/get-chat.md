# Chat Functions Documentation

Vector Vault provides two main chat functions for interacting with AI models: `get_chat()` for immediate responses and `get_chat_stream()` for real-time streaming responses. Both support features like conversation history, context-based responses through vector similarity search, custom prompting, and image processing capabilities.

## Function Signatures

### get_chat() - Standard Response
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
    timeout: int = 300,           # API timeout in seconds
    image_path: str = None,       # Path to local image file
    image_url: str = None,        # URL to image for processing
)
```

### get_chat_stream() - Streaming Response
```python
def get_chat_stream(
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
    metatag: bool = False,        # Use specific metadata tags
    metatag_prefixes: bool = False,  # Custom prefixes for metadata tags
    metatag_suffixes: bool = False,  # Custom suffixes for metadata tags
    custom_prompt: bool = False,   # Custom prompt template
    temperature: int = 0,         # Response randomness (0-1)
    timeout: int = 300,           # API timeout in seconds
    image_path: str = None,       # Path to local image file
    image_url: str = None,        # URL to image for processing
)
```

## Key Features

### Basic Response
Get a direct response from the AI model without additional context or features.
```python
# Standard response
response = vault.get_chat("What is Vector Vault?")

# Streaming response
for chunk in vault.get_chat_stream("What is Vector Vault?"):
    print(chunk, end='', flush=True)
```

### Conversation History
Include previous conversation context for more coherent responses.
```python
# Standard response with history
response = vault.get_chat("What about pricing?", history="User: What is Vector Vault?\nAI: Vector Vault is a vector database...")

# Streaming response with history
for chunk in vault.get_chat_stream("What about pricing?", history="Previous conversation..."):
    print(chunk, end='', flush=True)
```

### Text Summarization
Generate a concise summary of the input text.
```python
# Standard summarization
long_text = "Very long document content..."
summary = vault.get_chat(long_text, summary=True)

# Streaming summarization
for chunk in vault.get_chat_stream(long_text, summary=True):
    print(chunk, end='', flush=True)
```

### Context-Enhanced Responses (RAG)
Retrieve and utilize relevant context through vector similarity search.
```python
# Basic context-based response
response = vault.get_chat("How do I save data?", get_context=True)

# Context-based response with conversation history
response = vault.get_chat("What about loading?", history, get_context=True)

# Include context items in response
response = vault.get_chat("Explain the process", get_context=True, return_context=True)
print(response['response'])  # The AI response
print(response['context'])   # The context items used

# Streaming with context
for chunk in vault.get_chat_stream("How does it work?", get_context=True):
    print(chunk, end='', flush=True)
```

## Image Processing

Both functions support image processing capabilities for multimodal AI interactions.

```python
# Process local image
response = vault.get_chat(
    "What do you see in this image?",
    image_path="/path/to/image.jpg"
)

# Process image from URL
response = vault.get_chat(
    "Describe this image",
    image_url="https://example.com/image.jpg"
)

# Streaming image processing
for chunk in vault.get_chat_stream(
    "Analyze this image",
    image_path="/path/to/image.jpg"
):
    print(chunk, end='', flush=True)
```

## Streaming Utilities

![Vector Vault Streaming Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3FhcnB4MWEyeDdmNTRvNWVyODRoa3czMm9nM3RudDd5dW84Y3lwNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RAQQEzEZHjDwISYK8n/giphy.gif)

Vector Vault provides utility functions to handle streaming responses:

### print_stream() - Console Output
```python
# Pretty print streaming response to console
response = vault.print_stream(
    vault.get_chat_stream("Tell me about Vector Vault", get_context=True)
)
print(f"\nFinal response: {response}")
```

### cloud_stream() - Web Applications
```python
# For web applications (Flask, FastAPI, etc.)
@app.route('/chat')
def chat():
    return Response(
        vault.cloud_stream(
            vault.get_chat_stream("User message", get_context=True)
        ),
        mimetype='text/event-stream'
    )
```

## Advanced Features

### Custom Prompting
Custom prompts can be used to modify the AI's response style or behavior. Important notes:
- Only use custom prompts when `get_context=True`
- Custom prompts must include `{context}` and `{content}` placeholders
- For general prompt customization, use the personality message feature instead

```python
my_prompt = """Based on this context: {context}
Please answer this question: {content}
Respond in the style of a financial advisor."""

# Standard response
response = vault.get_chat("Should I invest?", get_context=True, custom_prompt=my_prompt)

# Streaming response
for chunk in vault.get_chat_stream("Should I invest?", get_context=True, custom_prompt=my_prompt):
    print(chunk, end='', flush=True)
```

### Smart History Search
Enable AI-powered search query generation based on conversation history to solve contextual search problems.

**The Problem**  In multi-turn conversations users often reply with highly contextual phrases like “Tell me more about that” or “Can you look that up?”  Taken on their own these phrases are meaningless for a vector similarity search and return irrelevant results.

**The Solution**  When `smart_history_search=True` the AI combines the *current* user message with the *conversation history* to generate a focused query that captures the real intent, then performs the vector search with that query.

```python
# Example conversation (pricing discussion)
history = """
User: Tell me about Vector Vault's pricing model
AI: Vector Vault offers flexible pricing based on usage…
User: What about enterprise features?
AI: Enterprise plans include advanced security and dedicated support…
"""

# User gives a vague follow-up question
after_message = "Ok, tell me more"

# WITHOUT smart_history_search – searches literally for the vague phrase
response = vault.get_chat(
    after_message,
    history,
    get_context=True
)  # Returns irrelevant results – search query is just "Ok, tell me more"

# WITH smart_history_search – AI rewrites the query using history
response = vault.get_chat(
    after_message,
    history,
    get_context=True,
    smart_history_search=True
)  # Vector search query becomes "Vector Vault enterprise advanced security features"
```

**When to Use**
• Multi-turn conversations where users reference earlier topics  
• Chatbots and interactive assistants  
• Anytime users employ pronouns or context-dependent phrases  
• Whenever precise context matching is required for relevant results

### Metadata Tag Control (Streaming Only)
Control how metadata is displayed in streaming responses:
```python
# Use specific metadata tags with custom formatting
response = vault.print_stream(
    vault.get_chat_stream(
        "Find documents about pricing",
        get_context=True,
        return_context=True,
        include_context_meta=True,
        metatag=['title', 'author'],
        metatag_prefixes=['\n\nTitle: ', '\nAuthor: '],
        metatag_suffixes=['', '\n']
    )
)
```

### Model Selection and Parameters
The AI system automatically handles model switching between platforms. Simply change the model parameter and Vector Vault will identify the correct platform (OpenAI, Claude, Grok, Groq, Gemini) and initialize it internally:

```python
# OpenAI model
response = vault.get_chat("Analyze this data", model="gpt-4o")

# Switch to Claude mid-conversation 
response = vault.get_chat("What's your opinion?", model="claude-sonnet-4-0")

# Switch to Grok
response = vault.get_chat("Be creative", model="grok-4")

# All in the same conversation - platform switching is automatic
for chunk in vault.get_chat_stream(
    "Creative writing task",
    model="gpt-4o",
    temperature=0.7,
    timeout=60
):
    print(chunk, end='', flush=True)
```

## Getting Started

1. **Initialize the Vault connection:**
```python
from vectorvault import Vault

vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_API_KEY',
    openai_key='YOUR_OPENAI_KEY',
    vault='your_vault_name'
)
```

2. **Make your first query:**
```python
# Standard response
question = "Why should I use Vector Vault for my next generative AI application?"
answer = vault.get_chat(question, get_context=True)
print(answer)

# Streaming response
question = "Why should I use Vector Vault for my next generative AI application?"
answer = vault.print_stream(
    vault.get_chat_stream(question, get_context=True)
)
```

## Error Handling and Rate Limiting

Both functions include built-in features for:
- Automatic retry logic with exponential backoff
- Rate limiting to prevent API abuse
- Token management for large inputs
- Graceful error handling with detailed logging
- Connection timeout management

## Best Practices

### General Guidelines
1. Use the personality message feature for general prompt customization instead of custom prompts
2. Enable `get_context=True` when domain-specific knowledge is required
3. **Always use `smart_history_search=True`** for multi-turn conversations and chatbots - it dramatically improves context relevance by generating better search queries from conversation history
4. Set appropriate `temperature` values based on your needs:
   - 0: Deterministic, focused responses
   - 0.5: Balanced creativity
   - 1: Maximum creativity
5. Adjust `timeout` based on expected response times and your application's needs (1-3 seconds for streaming responses is recommended since retry logic counts to the first token received)

### Streaming-Specific Guidelines
1. Use `get_chat_stream()` for real-time user interfaces where immediate feedback is important
2. Use `print_stream()` for console applications
3. Use `cloud_stream()` for web applications with server-sent events
4. Consider the trade-off between responsiveness and processing overhead
5. Handle streaming errors gracefully in your application

### Performance Considerations
1. Streaming responses provide better perceived performance for long responses
2. Standard responses are better for batch processing or when you need the complete response
3. Context-enhanced responses take longer but provide more accurate, relevant answers
4. Image processing requires additional time and may have different rate limits

## Example Use Cases

### Research Assistant
```python
# Research assistant for academic/scientific queries
def research_query(user_message, conversation_history):
    return vault.get_chat(
        user_message,
        history=conversation_history,
        get_context=True,
        smart_history_search=True,  # Critical for "Can you find more about that study?" queries
        temperature=0  # Deterministic, factual responses
    )

# Example where smart_history_search is essential:
# User: "Tell me about climate change effects on coral reefs"
# AI: "Climate change affects coral reefs through ocean acidification and temperature rise..."
# User: "What research supports that claim?"  
# 
# WITHOUT smart_history_search: Searches "what research supports that claim" → Generic research results
# WITH smart_history_search: Searches "climate change coral reef research studies ocean acidification temperature" → Specific relevant studies
```

### Customer Support Chatbot
```python
# Streaming customer support with context-aware search
def handle_support_query(user_message, conversation_history):
    return vault.print_stream(
        vault.get_chat_stream(
            user_message,
            history=conversation_history,
            get_context=True,
            smart_history_search=True,  # Essential for handling "What about that issue?" type queries
            model="gpt-4o"
        )
    )
```

### Real-time Chat Interface
```python
# Web streaming endpoint for real-time chat
@app.route('/stream-chat')
def stream_chat():
    user_message = request.json.get('message')
    history = request.json.get('history', '')
    
    return Response(
        vault.cloud_stream(
            vault.get_chat_stream(
                user_message,
                history=history,
                get_context=True,
                smart_history_search=True,  # Critical for web chat interfaces
                return_context=True
            )
        ),
        mimetype='text/event-stream'
    )
```

### Document Analysis with Images
```python
# Analyze document with image processing
def analyze_document(text_query, image_path, conversation_history=""):
    return vault.get_chat(
        text_query,
        history=conversation_history,
        image_path=image_path,
        get_context=True,
        smart_history_search=True,
        temperature=0  # Deterministic analysis
    )

# Example usage
result = analyze_document(
    "What are the key points in this financial report?",
    "/path/to/financial_report.pdf",
    "Previous discussion about Q3 earnings..."
)
```
