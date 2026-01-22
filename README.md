# Vector Vault

![Vector Vault Header](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

**Build AI agents that think, remember, and act.** Start free on your machine. Scale to production with zero infrastructure.

## Start Free, Scale When Ready

Vector Vault gives you two ways to build:

**🏠 Local Mode (Free)** — Run entirely on your machine. No account needed. No limits. Perfect for learning, prototyping, and projects where your data stays local.

**☁️ Cloud Platform** — When you're ready for production, deploy to our Persistent Agentic Runtime (PAR). Sub-second responses, 99.9% uptime, visual workflow builder, and agents that can pause for days and resume instantly.

```bash
pip install vector-vault
```

## Quick Start (Local Mode)

No signup. No API keys (except OpenAI for embeddings). Just code.

```python
from vectorvault import Vault

# Create a local vault
vault = Vault(
    vault='my_knowledge_base',
    openai_key='YOUR_OPENAI_KEY',
    local=True  # Everything stays on your machine
)

# Add your data
vault.add("The mitochondria is the powerhouse of the cell")
vault.add("Neural networks are inspired by biological brains")
vault.add("Vector databases enable semantic search")
vault.get_vectors()
vault.save()

# Search by meaning, not keywords
results = vault.get_similar("How do AI systems learn?")
# → Returns: "Neural networks are inspired by biological brains"

# Or chat with your data
response = vault.get_chat(
    "What powers the cell?",
    get_context=True  # Automatically retrieves relevant context
)
```

## What You Can Build

### RAG Applications
Give any LLM access to your knowledge base with automatic context retrieval.

```python
response = vault.get_chat(
    "How do I configure authentication?",
    get_context=True,
    n_context=5
)
```

### Semantic Search
Find content by meaning. Search "budget issues" and find documents about "financial constraints."

```python
results = vault.get_similar("budget issues", n=10)
```

### AI Memory Systems
Give your agents persistent memory across conversations.

```python
# Store conversation
vault.add(f"User asked about {topic}. Agent responded with {response}")
vault.get_vectors()
vault.save()

# Later, retrieve relevant context
context = vault.get_similar(new_user_message)
```

### Document Q&A
Turn any document collection into a question-answering system.

```python
# Load documents
for doc in documents:
    vault.add(doc.text, meta={'source': doc.filename})
vault.get_vectors()
vault.save()

# Answer questions
answer = vault.get_chat("What's the refund policy?", get_context=True)
```

## Going to Production

When you're ready to scale, Vector Vault Cloud provides:

### Persistent Agentic Runtime (PAR)
Agents that pause for days, branch into parallel tasks, and resume instantly — without you managing servers.

### Vector Flow
Design agent workflows visually with drag-and-drop. Branching logic, approvals, integrations, all in the browser.

### Production Performance
- Sub-second streaming responses
- 99.9% uptime SLA
- Auto-scaling to thousands of concurrent conversations

### Enterprise Ready
- SOC 2 compliant infrastructure
- Team collaboration
- Usage-based pricing

```python
# Switch to cloud mode
vault = Vault(
    user='you@company.com',
    api_key='YOUR_VECTORVAULT_KEY',
    openai_key='YOUR_OPENAI_KEY',
    vault='production_kb'
)

# Same API, production infrastructure
response = vault.get_chat("Customer question here", get_context=True)

# Or run visual workflows
result = vault.run_flow('customer_support_agent', user_message="...")
```

**[Get started at vectorvault.io →](https://vectorvault.io)**

## Core API

### Initialization

```python
# Local mode (free, no account)
vault = Vault(
    vault='vault_name',
    openai_key='sk-...',
    local=True
)

# Cloud mode (production)
vault = Vault(
    user='email',
    api_key='vv_...',
    openai_key='sk-...',
    vault='vault_name'
)
```

### Essential Methods

| Method | Description |
|--------|-------------|
| `add(text, meta=None)` | Add text to the vault |
| `get_vectors()` | Generate embeddings |
| `save()` | Persist to storage |
| `get_similar(text, n=4)` | Semantic search |
| `get_chat(text, get_context=True)` | RAG chat |
| `get_items(ids)` | Retrieve by ID |
| `edit_item(id, text)` | Update item |
| `delete_items(ids)` | Remove items |

### Convenience

```python
# Add + embed + save in one call
vault.add_n_save("Your text here")

# Stream responses
for chunk in vault.get_chat_stream("Your question"):
    print(chunk, end='')
```

## How It Works

Vector Vault uses FAISS (Facebook AI Similarity Search) for fast, accurate vector operations:

1. **Add** your text data
2. **Embed** using OpenAI's embedding models  
3. **Search** by semantic similarity
4. **Chat** with automatic context retrieval

Local mode stores everything in `~/.vectorvault/`. Cloud mode syncs to our managed infrastructure.

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings)

## Resources

- **Website**: [vectorvault.io](https://vectorvault.io)
- **Vector Flow**: [app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)
- **Full API Docs**: [Documentation](https://github.com/John-Rood/VectorVault/tree/main/vectorvault/documentation)
- **Discord**: [Join the community](https://discord.com/channels/1111817087007084544/1111817087451676835)
- **JavaScript SDK**: [VectorVault-js](https://github.com/John-Rood/VectorVault-js)

## Contributing

```bash
git clone https://github.com/John-Rood/VectorVault.git
cd VectorVault
pip install -e .

# Run tests
cd VectorVault-Testing
python run_tests.py        # Cloud tests
python test_local_mode.py  # Local tests
```

## License

MIT License

---

**Start free. Scale infinitely.** [vectorvault.io](https://vectorvault.io)
