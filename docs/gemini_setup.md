# Gemini Integration Setup Guide

This guide shows you how to set up and use Google Gemini models with VectorVault.

## Prerequisites

1. **Install the Google Generative AI library**:
   ```bash
   pip install google-generativeai
   ```

2. **Get your Gemini API key**:
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in your code

## Setup:

### Pass API Key Directly
```python
from vectorvault import Vault

vault = Vault(
    user='your_email@example.com',
    api_key='your_vectorvault_api_key',
    gemini_key='your_gemini_api_key',  # Pass key directly
    vault='your_vault_name'
)
```

## Usage Examples

### Basic Text Generation
```python
# Simple chat
response = vault.get_chat('Explain quantum physics', model='gemini-2.5-flash')
print(response)

# Streaming response
for chunk in vault.get_chat_stream('Tell me a story', model='gemini-2.5-pro', stream=True):
    print(chunk, end='')
```

### RAG (Retrieval Augmented Generation)
```python
# Add data to your vault first
vault.add('Quantum computers use quantum mechanics principles...')
vault.add('Machine learning algorithms can recognize patterns...')
vault.save()

# Use context from your vault
response = vault.get_chat(
    'What are quantum computers?', 
    get_context=True,  # This retrieves relevant context from your vault
    model='gemini-2.5-flash'
)
```

### Image Analysis
```python
# Analyze an image
response = vault.ai.image_inference(
    image_path='path/to/your/image.jpg',
    user_text='What do you see in this image?',
    model='gemini-2.5-flash'
)
print(response)

# Analyze image from URL
response = vault.ai.image_inference(
    image_url='https://example.com/image.jpg',
    user_text='Describe this image in detail',
    model='gemini-2.5-pro'
)
```

### Advanced Configuration
```python
# Custom prompts and personality
vault = Vault(
    user='your_email@example.com',
    api_key='your_vectorvault_api_key',
    gemini_key='your_gemini_api_key',
    vault='your_vault_name',
    main_prompt="Answer this question: {content}",
    personality_message="Be concise and technical",
    verbose=True
)

# Custom model with fine-tuning
vault.add_fine_tuned_model_to_platform(
    model_name='my-custom-gemini-model',
    platform='gemini',
    token_limit=500000
)
```

## Available Models

| Model | Token Limit | Best For |
|-------|-------------|----------|
| `gemini-2.5-pro` | 1M+ tokens | Complex reasoning, coding, analysis |
| `gemini-2.5-flash` | 1M+ tokens | Fast responses, general tasks (default) |
| `gemini-2.5-flash-lite` | 1M+ tokens | Lightweight tasks |
| `gemini-2.0-flash` | 1M+ tokens | Multimodal tasks |

## Troubleshooting

### Error: "Missing key inputs argument"
This means your API key is not properly set. Try:

1. **Check your API key**:
   ```python
   # Test your key directly
   from google import genai
   client = genai.Client(api_key='your_api_key')
   response = client.models.generate_content(
       model='gemini-2.5-flash',
       contents='Hello, world!'
   )
   print(response.text)
   ```
   
3. **Pass key explicitly**:
   ```python
   vault = Vault(gemini_key='your_api_key', ...)
   ```

### Error: "Model not found"
Make sure you're using one of the supported models listed above.

### Error: "Client not initialized"
This means the Gemini client couldn't be created. Check your API key and internet connection.

## Features

- ✅ Text generation (streaming & non-streaming)
- ✅ Multimodal image processing
- ✅ Accurate token counting using Gemini API
- ✅ Automatic model switching based on token limits
- ✅ Full integration with VectorVault's RAG system
- ✅ Support for custom prompts and personalities
- ✅ Thinking capabilities (built into 2.5 series models)

## Example: Complete Workflow

```python
from vectorvault import Vault

# Initialize vault with Gemini
vault = Vault(
    user='your_email@example.com',
    api_key='your_vectorvault_api_key',
    gemini_key='your_gemini_api_key',
    vault='my_knowledge_base'
)

# Add knowledge to your vault
vault.add('Python is a programming language...')
vault.add('Machine learning uses algorithms...')
vault.save()

# Get AI responses with context
response = vault.get_chat(
    'How can I use Python for machine learning?',
    get_context=True,
    model='gemini-2.5-flash'
)
print(response)

# Analyze an image
image_response = vault.ai.image_inference(
    image_path='code_screenshot.png',
    user_text='Explain this code',
    model='gemini-2.5-pro'
)
print(image_response)
```

That's it! You're now ready to use Gemini models with VectorVault. 🚀 