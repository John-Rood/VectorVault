# Vector Flow Basics: Building AI Agents with No Code

Vector Flow is a visual development environment where you build sophisticated AI agents through drag-and-drop composition. Design complex reasoning patterns, multi-step workflows, and intelligent decision trees that deploy instantly to production-grade infrastructure.

Every flow you create runs on Vector Vault's Persistent Agentic Runtime (PAR), delivering production-ready performance: sub-second response times, thousands of concurrent conversations, and state that persists across days or weeks of execution. What you design visually becomes a high-performance agent without any additional engineering work.

## Table of Contents

1. [What Makes Vector Flow Useful](#what-makes-vector-flow-useful)
   - [Key Technical Advantages](#key-technical-advantages)
2. [Getting Started: Your First Flow](#getting-started-your-first-flow)
   - [Creating a New Flow](#creating-a-new-flow)
   - [Building Your Flow: The Canvas](#building-your-flow-the-canvas)
   - [Testing Your Flow: Real-Time Development](#testing-your-flow-real-time-development)
3. [Your First Flow: The Dog Expert](#your-first-flow-the-dog-expert)
   - [Flow Diagram](#flow-diagram)
   - [Step-by-Step Build](#step-by-step-build)
   - [Example Conversation](#example-conversation)
   - [What You Just Learned](#what-you-just-learned)
4. [Core Node Types: Your Building Blocks](#core-node-types-your-building-blocks)
   - [AI Intelligence Nodes](#ai-intelligence-nodes)
   - [Action & Integration Nodes](#action--integration-nodes)
   - [Data & Storage Nodes](#data--storage-nodes)
   - [Control Flow Nodes](#control-flow-nodes)
5. [Understanding Variables: The Data Flow](#understanding-variables-the-data-flow)
   - [Setting Variables](#setting-variables)
   - [Using Variables](#using-variables)
   - [Variable Scope](#variable-scope)
6. [Building Patterns: From Simple to Complex](#building-patterns-from-simple-to-complex)
   - [Simple Pattern](#simple-pattern-question--recognition--response)
   - [Intermediate Pattern](#intermediate-pattern-data-collection--processing--action)
   - [Advanced Pattern](#advanced-pattern-parallel-recognition--conditional-logic)
7. [Advanced Concepts: Scaling Your Agents](#advanced-concepts-scaling-your-agents)
   - [Child Flows for Modularity](#child-flows-for-modularity)
   - [Persistent State](#persistent-state)
   - [Performance Optimization](#performance-optimization)
8. [Next Steps: From Flow to Production](#next-steps-from-flow-to-production)
   - [Integration Example](#integration-example)
9. [Ready to Build?](#ready-to-build)

## Related Documentation

- [Advanced Node Reference](advanced-features.md) - Deep dive into advanced node configurations
- [Vector Flow Logging](logging.md) - Understanding flow execution logs and debugging
- [Python Integration & Backend Runtime](python-integration.md) - Server-side integration and deployment
- [JavaScript Integration](https://github.com/John-Rood/VectorVault-js/blob/main/vectorflow_docs.md) - Client-side integration with Vector Flow

---

## Why Visual Agent Development Works

Visual development accelerates agent creation by making complex behaviors immediately understandable and testable. When you can see your agent's decision tree, trace execution paths in real-time, and modify logic without redeploying code, iteration speed increases dramatically.

Vector Flow provides this visual layer on top of production infrastructure, giving you the best of both worlds:

### Execution Layer (What You Don't Build)
- **State persistence**: Automatic checkpointing between nodes
- **Concurrency control**: Safe parallel execution with shared state
- **Error recovery**: Built-in retries with exponential backoff
- **Resource management**: Automatic scaling and load balancing
- **Performance monitoring**: Request tracing and latency metrics

### What You Actually Build
- **Recognition patterns**: "Is this a refund request?" → Yes/No routing
- **Data flows**: Extract customer ID → Query database → Generate response
- **Conditional logic**: If premium customer AND issue urgent → Escalate
- **Integrations**: Call your APIs, query vaults, send emails

## Architecture That Scales

Every Vector Flow runs on infrastructure designed for enterprise workloads:

```
Your Flow Design                     PAR Runtime
┌─────────────────┐                 ┌──────────────────┐
│ Recognition Node│    Compiles     │ Stateless Worker │
│       ↓         │        →        │   Recognition    │
│ Parallel Branch │                 │   ↓         ↓    │
│   ↓         ↓   │                 │ State  Checkpoint│
│ API Call  Email │                 │   ↓         ↓    │
└─────────────────┘                 │ Continue    Save │
                                    └──────────────────┘
```

### Performance Characteristics
Based on millions of production executions:

- **Recognition latency**: p50 380ms, p99 950ms
- **State operations**: ~5,000/sec reads, ~1,000/sec writes
- **Concurrent flows**: 10,000+ per project without degradation
- **Uptime**: 99.95% over past 12 months

**Real-world example**: Aegis AI runs their "AI Firewall" security platform on Vector Flow as the decision layer, processing thousands of threat assessments per day without rewriting their prototype.

## Getting Started:

### Creating a New Flow

1. **Start Fresh**: Click the "+" button to create a new flow, or duplicate an existing flow as a template
2. **Every Flow Needs a Start**: Drag the "Start" node (▶️ play icon) from the sidebar—this is your flow's entry point
3. **Access Defaults**: Click the chevron (>) in the top-left corner next to the node sidebar to set global defaults (model, vault, etc.) that apply to all nodes unless overridden

### Building Your Flow: The Canvas

The flow builder uses a visual canvas where you drag, connect, and configure nodes:

- **Drag Nodes**: Pull node types from the left sidebar onto the canvas
- **Connect Nodes**: Draw arrows between node outputs and inputs to define the flow path
- **Configure Nodes**: Click any node to edit its settings in the right panel
- **Delete Nodes**: Right-click and select delete
- **Use Variables**: Type `{variable_name}` in any text field to inject dynamic data

### Testing Your Flow: Real-Time Development

Click "Open Chat" to test your flow as you build:

- **Left Side**: Chat interface where you interact with your AI agent
- **Right Side**: Live logs showing exactly what each node is doing, thinking, and outputting
- **Real-Time Debugging**: See what goes wrong and fix it immediately by adjusting prompts and logic

This dual-pane setup makes it easy to inspect what each node is doing and adjust prompts quickly. For detailed information about the logging system, see the [Vector Flow Logging documentation](logging.md).

## Your First Flow: The Dog Expert

Let's build a simple example: an AI agent that only talks about dogs. This demonstrates the core concepts of recognition, routing, and responses.

### Flow Diagram
```
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│  START  │────▶│  RECOGNIZE  │────▶│   RESPOND    │
└─────────┘     │"Is this     │ YES │              │
                │about a dog?"│     └──────────────┘
                └─────────────┘
                       │ NO
                       ▼
                ┌──────────────┐
                │   RESPOND    │
                │"I only talk  │
                │about dogs"   │
                └──────────────┘
```

### Step-by-Step Build

| Step | Action | Details |
|------|--------|---------|
| 1 | **Add START node** | Drag the START node (▶️) from sidebar to canvas |
| 2 | **Add RECOGNIZE node** | Drag RECOGNIZE node, connect START → RECOGNIZE |
| 3 | **Configure Recognition** | Click RECOGNIZE, set prompt: `"Is this about a dog?"` |
| 4 | **Add First RESPOND** | Drag RESPOND node, connect RECOGNIZE "Yes" → RESPOND |
| 5 | **Configure Dog Response** | Optional: Set prompt like `"Answer helpfully about dogs"` (or leave blank for automatic responses) |
| 6 | **Add Second RESPOND** | Drag another RESPOND node, connect RECOGNIZE "No" → this RESPOND |
| 7 | **Configure Rejection** | Set prompt: `"Tell them you only talk about dogs"` |
| 8 | **Save Your Flow** | Click Save, name it "Dog Expert" |
| 9 | **Test It!** | Click "Open Chat" and try both dog and non-dog questions |

### Example Conversation
```
User: "Tell me about Golden Retrievers"
Bot: "Golden Retrievers are wonderful, friendly dogs known for their intelligence and loyalty..."

User: "What's the weather like?"
Bot: "I only talk about dogs! Is there anything about dogs I can help you with?"
```

### What You Just Learned
- ✅ **Flow Logic**: How nodes connect to create decision paths
- ✅ **Recognition**: AI making fast yes/no routing decisions  
- ✅ **Conditional Responses**: Different outputs based on AI recognition
- ✅ **Prompting**: When to guide AI responses vs. letting them be automatic
- ✅ **Testing**: Real-time chat interface for immediate feedback

This simple flow highlights how recognition nodes route users to different RESPOND nodes, giving you conditional logic without writing code.

## Core Node Types: Your Building Blocks

Vector Flow provides specialized nodes for different aspects of AI agent behavior:

### AI Intelligence Nodes
- **RESPOND**: Generate AI responses to users (supports streaming)
- **RECOGNIZE**: Fast yes/no AI decisions for routing and logic
- **GENERATE**: Create text and store it in variables for later use
- **CAPTURE**: Extract specific information (names, emails, etc.) from conversations
- **MULTIPLE CHOICE**: Categorize inputs and route to different paths
- **NUMBER**: Generate integers for scoring, counting, or mathematical logic

### Action & Integration Nodes
- **ACT**: Write Python code for API calls, data processing, or custom logic - [ACT Node Documentation](nodes/act-node.md)
- **EMAIL**: Send automated emails with dynamic content
- **DOWNLOAD URL**: Fetch content from websites
- **GOOGLE SEARCH**: Perform searches and use results in your flow
- **MCP SERVER**: Connect to external tools and services - [MCP Node Documentation](nodes/mcp-node.md)

### Data & Storage Nodes
- **ADD TO VAULT**: Store information in your vector database
- **STORAGE**: Get, set, or delete data with path-based organization
- **VARIABLE**: Set static values for use throughout the flow

### Control Flow Nodes
- **IF/THEN**: Create conditional logic based on variables or AI decisions
- **RUN FLOW**: Call other flows as sub-processes (they share state with the parent)
- **PARALLEL**: Execute multiple paths simultaneously, then continue when all complete

## Understanding Variables: The Data Flow

Variables are how data moves through your flow. They're the "memory" that makes your agent intelligent and contextual.

### Setting Variables
Most nodes can "Set Variable" in their configuration, storing their output for later use:
- **CAPTURE** extracts multiple variables at once (e.g., `{first_name}`, `{email}`)
- **GENERATE** creates one variable with AI-generated content
- **RESPOND** can save the full response after streaming
- **ACT** sets variables via Python: `save['customer_data'] = api_response` - [ACT Node Documentation](nodes/act-node.md)

### Using Variables
Reference any variable in text fields with bracket notation: `{variable_name}`
- In prompts: "Hello {first_name}, your order {order_id} is ready"
- In Python (ACT nodes): Access directly like `print(first_name)` (no brackets needed)
- In conditions: Use in IF/THEN logic for dynamic branching

### Variable Scope
Variables persist throughout the entire flow execution and are automatically shared with child flows, enabling complex multi-flow architectures.

## Building Patterns: From Simple to Complex

### Simple Pattern: Question → Recognition → Response
```
Start → Recognize ("Is this a support question?") 
      ├── Yes → Respond ("I'll help you with that...")
      └── No → Respond ("Let me transfer you to sales...")
```

### Intermediate Pattern: Data Collection → Processing → Action
```
Start → Capture (extract {name}, {email}, {issue}) 
      → ACT (lookup customer in database via API)
      → Generate (create personalized response using API data)
      → Email (send follow-up with {name} and issue details)
```

### Advanced Pattern: Parallel Recognition → Conditional Logic
```
Start → Parallel → [Multiple Recognize nodes running simultaneously]
                  ├── Recognize ("Urgent?") → {is_urgent}
                  ├── Recognize ("Technical?") → {is_technical}  
                  └── Recognize ("Existing customer?") → {is_customer}
                  
      → IF/THEN (check {is_urgent} AND {is_technical})
         ├── True → Run Flow ("escalate_to_expert")
         └── False → Standard response flow
```

## Advanced Concepts: Scaling Your Agents

### Child Flows for Modularity
The RUN FLOW node calls other flows as sub-processes. Child flows:
- Share all variables with the parent flow
- Can run in parallel for exponential processing power
- Help break complex logic into maintainable modules
- Enable reusable agent components across multiple flows

### Persistent State
Your flows maintain state across any runtime interruption—containers, wait periods, or system restarts. This means you can build complex, long-running processes without worrying about losing context.

### Testing Strategy
1. **Unit test nodes**: Use test panel to verify each node independently
2. **Flow coverage**: Ensure all paths tested with example inputs
3. **Load testing**: Built-in simulator can run 1,000 concurrent conversations
4. **A/B testing**: Deploy multiple versions with traffic splitting

### Monitoring & Observability
Every flow automatically tracks:
- **Latency percentiles**: p50, p95, p99 by node type
- **Error rates**: With automatic alerting above thresholds
- **Token usage**: Cost attribution by flow and customer
- **Conversation analytics**: Sentiment, topics, outcomes

Access via dashboard or export to your monitoring stack (Datadog, NewRelic, CloudWatch).

1. **Deploy as Widget**: Create embeddable chat widgets for websites
2. **Programmatic Integration**: Call flows from Python/JS applications with `vault.run_flow()` or `vault.stream_flow()` (see [Backend Runtime & Integration](python-integration.md) and [JavaScript Integration](https://github.com/John-Rood/VectorVault-js/blob/main/vectorflow_docs.md) docs)
3. **Advanced Features**: Explore model selection, context retrieval, and temperature controls (see [Advanced Flow Builder Features](advanced-features.md))

### Integration Example
```python
# Your agent becomes part of your application
response = vault.run_flow(
    flow_name='customer_support_agent',
    message=user_input,
    customer_id=customer_id,
    tier="premium"
)
```

For complete integration examples and API reference, see the [Backend Runtime & Integration documentation](python-integration.md).

Vector Flow handles orchestration so you can focus on prompts, data sources, and human-in-the-loop steps.

## Ready to Build?

Head to **vectorvault.io/vector-flow** and start with a simple Start → Recognize → Respond flow. Use the chat interface to test, watch the logs to debug, and iterate until your agent behaves exactly as intended. Then scale up to more complex patterns as you get comfortable with the platform.

Remember: Every production agent started as a simple flow. The key is starting with one specific use case, testing with real users, and iterating based on what you learn. Vector Flow handles the infrastructure complexity so you can focus on what matters—building agents that solve real problems.