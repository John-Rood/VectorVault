# Vector Vault

![Vector Vault Header](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is an AI development platform for building, deploying, and operating autonomous agents with persistent memory. It combines a managed vector database, multi-model orchestration, and a flow-based runtime so teams can design long-running workflows without stitching together separate services.

## Overview

- Build state-aware agents that resume work after pauses or scheduled triggers.
- Compose workflows visually in Vector Flow or programmatically through the Python SDK.
- Run retrieval augmented generation (RAG) across one or many vaults with a consistent API.
- Stream responses, capture logs, and deploy to cloud-hosted runtimes without custom infrastructure.

### Vector Flow: Visual Agent Construction
Vector Flow (available at [app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)) is the visual builder for assembling agents. It provides:

- Drag-and-drop node editing with recognition, response, and tool nodes.
- Multi-provider AI connections (OpenAI, Anthropic, Groq, Grok, Gemini).
- Secure Python execution inside flows for API calls or custom logic.
- Built-in deployment so the same flow can run in development or production.

### Persistent Agentic Runtime (PAR)
Vector Flow runs on the Persistent Agentic Runtime. Application compute stays serverless and stateless, while each agent stores conversation state, variables, and checkpoints in the cloud. As a result:

- Agents resume exactly where they paused, without rebuilding context.
- Long-running flows can wait on external events for minutes or days.
- Stateless workers scale up on-demand while state remains durable.

## ⚡ Quick Start: Your First Autonomous Agent

#### Install:
```bash
pip install vector-vault
```

#### Build an Intelligent Agent in Minutes:

```python
from vectorvault import Vault

# Initialize with multi-platform AI support
vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_VECTOR_VAULT_API_KEY', 
    openai_key='YOUR_OPENAI_API_KEY',
    anthropic_key='YOUR_ANTHROPIC_KEY',  # optional
    vault='MY_AGENT_VAULT'
)

# Build your agent's knowledge base
vault.add('Your domain expertise, technical docs, and procedures...')
vault.get_vectors()
vault.save()

# Deploy autonomous workflows
agent_response = vault.run_flow(
    'intelligent_assistant',
    'Process this new customer inquiry',
    customer_data={"tier": "premium", "history": [...]},
    escalation_rules={"urgent": True}
)

# Get context-aware responses with smart history
response = vault.get_chat(
    "What about that issue we discussed earlier?",
    history=conversation_history,
    get_context=True,
    smart_history_search=True,  # AI generates contextual search queries
    model="claude-sonnet-4-0"   # Switch models seamlessly
)
```

## Multi-Provider Chat Interfaces

Vector Vault routes chat and flow requests to OpenAI, Anthropic, Groq, Grok, and Gemini through one interface. Each call can target a different model, so you can mix providers inside a single workflow without rewriting prompts or transport code:

```python
# Start with OpenAI for analysis
response = vault.get_chat("Analyze this data", model="gpt-4o")

# Switch to Claude for reasoning  
response = vault.get_chat("What's your recommendation based on that?", model="claude-sonnet-4-0")

# Use Grok for creative tasks
response = vault.get_chat("Now, generate some innovative solutions", model="grok-4")
```

## Smart History Search

The `smart_history_search` option asks the model to rewrite vague follow-ups ("How do I fix that?") into explicit search terms before querying the vector store. This keeps RAG grounded in the full conversation rather than only the latest user message.

```python
# User: "I'm getting database timeout errors in PostgreSQL"
# AI: "Here are some common causes..."
# User: "How do I fix that?"

# WITHOUT smart search: Searches "how do I fix that" → returns random, generic results
# WITH smart search: Searches "PostgreSQL database timeout errors fix" → returns specific solutions

response = vault.get_chat(
    "How do I fix that?",  # Vague, contextual query
    history=conversation_history,
    get_context=True,
    smart_history_search=True
)
```

## Additional Capabilities

### Multimodal Intelligence
Build agents that can see and understand images and documents.
```python
response = vault.get_chat(
    "Analyze the key takeaways from this financial report",
    image_path="/path/to/report.pdf", 
    get_context=True
)
```

### Real-Time Streaming
Stream tokens to consoles or web clients instead of waiting for a full response.
```python
# Console applications
response = vault.print_stream(
    vault.get_chat_stream("Research the latest AI trends", get_context=True)
)

# Web applications (Server-Sent Events)
@app.route('/agent-stream')
def agent_chat():
    return Response(
        vault.cloud_stream(vault.get_chat_stream(user_message, get_context=True)),
        mimetype='text/event-stream'
    )
```

- **Context payload behavior**:
  - **get_chat**: if you pass `return_context=True`, it returns a dict: `{'response': str, 'context': list}`.
  - **get_chat_stream**: if you pass `return_context=True`, after token streaming finishes, a single JSON string is yielded just before `!END` with the same shape: `{"response": "<full_response>", "context": [...]}`.
  - **Inline context streaming** happens only when you also provide metatag parameters.

- **How to catch the final context payload (server-side streaming to frontend)**:
```python
# Capture JSON context payload (do not append or emit)
if isinstance(word, str) and word.startswith('{"response":') and word.endswith('}'):
    try:
        context_data = json.loads(word)
        collected_context = context_data.get('context')
        continue
    except Exception:
        pass
else:
    # handle normal token output
    ...
```

### Cross-Vault Retrieval
Search multiple vaults at once. Each query fans out, collects candidates, and returns a single sorted list so downstream code can treat the result like any other search.

```python
# Chat with cross-vault context
response = vault.get_chat(
    "Where is the refund policy and API rate limits documented?",
    get_context=True,
    n_context=6,
    vaults=["docs", "customer_support", "developer_portal"]
)

# Or use the low-level API directly
results = vault.get_similar_from_vaults(
    "refund policy",
    n=8,
    vaults=["docs", "customer_support", "legal"]
)

# Enforce per-vault minimums (remaining slots fill from best overall)
results = vault.get_similar_from_vaults(
    "refund policy",
    n=5,
    vaults={"legal": 2, "docs": 1}
)
```

Notes:
- You can pass a single vault as a string to target only that vault
- You can enforce per-vault minimums with a dict: `vaults={"legal": 2, "docs": 1}`; if sum(minima) > n, n automatically expands to the sum of minima; otherwise the remainder fills from best overall

## When to Use Vector Vault

- Build autonomous agents that need durable memory, scheduling, or multi-step flows.
- Share the same vaults across SDK calls, Vector Flow, and cloud-hosted runtimes.
- Mix LLM providers or models mid-conversation without rebuilding your stack.
- Support multi-tenant products with isolated vaults and cross-vault retrieval.

## Resources

### Get Started
1. **30-day free trial**: [VectorVault.io](https://vectorvault.io)
2. **Visual agent builder**: [app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)  
3. **Install the platform**: `pip install vector-vault`

### Learn More
- **Full Documentation**: [API Reference](https://github.com/John-Rood/VectorVault/tree/main/vectorvault/documentation/fulldocs.md)
- **Vector Flow Guide**: [Agent Building Documentation](https://github.com/John-Rood/VectorVault/tree/main/documentation/vectoflow_docs)
- **Chat Functions**: [RAG & Streaming Guide](https://github.com/John-Rood/VectorVault/tree/main/vectorvault/documentation/get_chat_docs.md)
- **Community**: [Discord](https://discord.com/channels/1111817087007084544/1111817087451676835)
- **JavaScript SDK**: [VectorVault-js](https://github.com/John-Rood/VectorVault-js)

**Questions?** Open an issue or join the Discord community—feedback is welcome.
