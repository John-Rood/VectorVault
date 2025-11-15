# Vector Vault

![Vector Vault Header](https://images.squarespace-cdn.com/content/646ad2edeaaf682a9bbc36da/297fde6c-f5b4-4076-83bc-81dcfdbffebe/Vector+Vault+Header+5000.jpg)

Vector Vault is a hosted platform for building and running AI agents with persistent memory. You design agent workflows visually in the Vector Flow browser app, and we handle the infrastructure - delivering sub-second response times and scaling to thousands of concurrent operations without you managing servers or databases.

The platform centers on PAR (Persistent Agentic Runtime), which separates ephemeral compute from durable state. This architecture enables agents that pause for days waiting on approvals, branch into hundreds of parallel tasks, or maintain conversational context across months—all while keeping your operational overhead minimal.

## What Makes This Different

Traditional agent frameworks require you to wire together vector databases, model routers, state stores, and orchestration layers. Even then, you're left optimizing for latency, managing concurrency, and dealing with infrastructure scaling. Vector Vault consolidates these into a managed runtime where:

- **Performance is built-in**: Sub-second streaming responses and ~1,000 writes/sec throughput come standard, not after months of optimization.
- **State persists automatically**: Agents resume exactly where they left off, whether after 5 minutes or 5 days.
- **Scaling is transparent**: Handle 10 or 10,000 concurrent conversations without changing your architecture.
- **Costs stay predictable**: Pay for what you use, not for idle infrastructure or over-provisioned clusters.

## Core Architecture

### Vector Flow: Visual Agent Design
[Vector Flow](https://app.vectorvault.io/vector-flow) is where you build agents through drag-and-drop composition. Behind its accessible interface lies a sophisticated execution engine:

- **Smart routing via recognition nodes**: Binary decisions execute in ~300ms, enabling real-time conversation flow.
- **Native parallelism**: Child flows inherit parent state and run concurrently, allowing exponential scaling patterns.
- **Python-in-the-browser**: The Act node provides a sandboxed Python environment for API integrations and data processing.
- **Multi-model orchestration**: Switch between OpenAI, Claude, Gemini, and Grok mid-conversation based on task requirements.

### Persistent Agentic Runtime (PAR)
PAR is the technical foundation that makes Vector Vault agents different:

```
User Message → Stateless Worker → PAR State Layer → Response
                     ↑                    ↓
                     └──── Durable State ────┘
```

This separation means:
- **Workers scale elastically**: New instances spin up in milliseconds to handle load spikes.
- **State remains consistent**: Variables, conversation history, and checkpoints persist across any number of workers.
- **Long-running flows become trivial**: An agent can wait weeks for a human approval without consuming compute resources.

Performance characteristics (typical production workloads):
- **Latency**: 200-400ms for vector retrieval + model generation
- **Throughput**: ~5,000 reads/sec and ~1,000 writes/sec per project
- **Availability**: 99.9% uptime SLA with automatic failover

## 🚀 Quick Start

```bash
pip install vector-vault
```

### Your First Agent in 60 Seconds

```python
from vectorvault import Vault

# Connect to Vector Vault
vault = Vault(
    user='YOUR_EMAIL',
    api_key='YOUR_VECTOR_VAULT_API_KEY', 
    openai_key='YOUR_OPENAI_KEY',
    anthropic_key='YOUR_ANTHROPIC_KEY',  # optional - for Claude models
    vault='customer_support'
)

# Add your knowledge base
vault.add("""
    Refund policy: Full refund within 30 days...
    Shipping: Standard 5-7 business days, Express 1-2 days...
    Customer service hours: 24/7 via chat, 9-5 EST via phone...
""")
vault.get_vectors()
vault.save()

# Deploy a simple customer service flow
response = vault.run_flow(
    'customer_service_agent',
    user_message='I ordered 5 days ago but haven't received tracking',
    customer_data={'order_date': '2024-01-10', 'shipping': 'standard'}
)
```

## Advanced Capabilities

### Intelligent Context Management
Our Smart History Search transforms vague follow-ups into precise retrieval queries:

```python
# Conversation context
history = """
User: My PostgreSQL queries are timing out after 30 seconds
AI: That's likely due to missing indexes or table locks...
User: How do I fix that?
"""



response = vault.get_chat(
    "How do I fix that?",
    history=history,
    get_context=True,
    smart_history_search=True
)
```
Smart History automatically reformulates the user input "How do I fix that?" 
into a search query "PostgreSQL query timeout solutions indexes locks"
So, instead of searching the database for "How do I fix that", 
We search the database for "PostgreSQL query timeout solutions indexes locks"
Which yields the right context for answering the question

### Cross-Vault Intelligence
Search across isolated data repositories while maintaining security boundaries:

```python
# Simultaneously search documentation, support tickets, and legal policies
results = vault.get_similar_from_vaults(
    "GDPR compliance for user data deletion",
    n=10,
    vaults={
        "legal_docs": 3,      # Guarantee 3 results from legal
        "support_tickets": 2,  # 2 from support history
        "engineering": None    # Fill remaining slots with best engineering docs
    }
)
```

### Production-Ready Streaming
Built for real-time applications with proper error handling and backpressure:

```python
# Stream to web clients with automatic chunking and keep-alive
@app.route('/chat-stream')
def chat_stream():
    def generate():
        for token in vault.get_chat_stream(request.json['message'], 
                                          get_context=True,
                                          model='claude-4-5-sonnet'):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

## Real-World Applications

### Customer Support Automation
A major e-commerce platform replaced their traditional chatbot with Vector Vault agents:
- **Response quality**: 73% → 89% first-contact resolution
- **Latency**: 2.3s → 0.4s average response time
- **Scale**: Handles 50K+ concurrent conversations during peak sales

### AI-Powered Security Platform
Aegis AI built their entire threat detection system on Vector Flow:
- **Processing volume**: 100K+ security events daily
- **Decision speed**: <300ms for threat classification
- **Zero downtime**: Since deployment 8 months ago

### Healthcare Intake Automation
A telehealth startup uses Vector Vault for patient screening:
- **Compliance**: HIPAA-compliant isolated vaults per patient
- **Accuracy**: 94% correct triage decisions
- **Time savings**: 15-minute manual intake → 3-minute automated flow

## When Vector Vault Makes Sense

**Perfect fit:**
- You need agents that maintain state across long time periods
- Your use case involves complex, multi-step workflows
- You want production-grade performance without infrastructure work
- You need to mix different AI models based on task requirements

**Consider alternatives if:**
- You want to run local models
- You require on-premise deployment for regulatory reasons

## Pricing Philosophy

We charge for actual usage, not reserved capacity:
- **No minimum commits**: Start with $0, scale as you grow
- **Transparent metering**: Pay per operation (embeddings, searches, model calls)
- **Volume discounts**: Automatic price breaks at scale
- **Predictable bills**: Hard spending caps available

Most teams find they spend 40-70% less than running equivalent infrastructure themselves, while getting better performance.

## Get Started

1. **Try it free**: [VectorVault.io](https://vectorvault.io) - 30-day trial includes all features
2. **Build visually**: [app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)
3. **Install the SDK**: `pip install vector-vault`

### Resources
- **Complete API Docs**: [Full API Reference](https://github.com/John-Rood/VectorVault/tree/main/vectorvault/documentation/fulldocs.md)
- **Vector Flow Guide**: [Visual Agent Building](https://github.com/John-Rood/VectorVault/tree/main/documentation/vectoflow_docs)
- **Architecture Deep Dive**: [How PAR Works](https://github.com/John-Rood/VectorVault/tree/main/vectorvault/documentation/get_chat_docs.md)
- **Discord Community**: [Join 2,000+ builders](https://discord.com/channels/1111817087007084544/1111817087451676835)
- **JavaScript SDK**: [Browser & Node.js](https://github.com/John-Rood/VectorVault-js)

---
