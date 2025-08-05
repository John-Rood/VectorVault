# Vector Flow Basics: Building AI Agents with No Code



Vector Flow is your **Agent Operating System**—a visual, no-code platform for building sophisticated AI agents that can think, reason, and act. Whether you're creating an AI firewall (like Aegis AI), a customer support agent, a sales automation system, or complex backend AI workflows, Vector Flow handles the orchestration so you can focus on building.

## Table of Contents

1. [What Makes Vector Flow Powerful](#what-makes-vector-flow-powerful)
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

- [Advanced Flow Builder Features](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_advanced_features.md) - Deep dive into advanced node configurations
- [Vector Flow Logging](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_logging.md) - Understanding flow execution logs and debugging
- [Python Integration & Backend Runtime](https://github.com/John-Rood/VectorVault/tree/main/vectoflow_docs/vectoflow_python.md) - Server-side integration and deployment
- [JavaScript Integration](https://github.com/John-Rood/VectorVault-js/blob/main/vectorflow_docs.md) - Client-side integration with Vector Flow

---

## What Makes Vector Flow Powerful

Vector Flow is your **Agent Operating System** for building sophisticated AI that thinks, reasons, and acts. Here's what you can do:

### High-Level Overview: What You Can Do with Vector Flow

- **Build Agents**: Create AI agents for conversation handling, data extraction, decision-making, and automation
- **Orchestrate LLMs**: Combine AI nodes for generation, recognition, and logic without coding
- **Integrate Data**: Use vaults (vector databases) for semantic search and RAG (Retrieval-Augmented Generation)
- **Run Flexibly**: Test in the builder, deploy as widgets, or execute via APIs in your apps
- **Scale with Child Flows**: Nest flows for modularity, sharing state and enabling exponential parallel processing

Whether you're automating customer support or building backend AI for software, Vector Flow is the backend runtime that lets you build without limits.

### Advanced Capabilities

- **Long-Running Agents**: Build agents that literally never stop running—perfect for monitoring, automation, and continuous processing. Use the agent control dashboard at **vectorvault.io/agents** with full observability through runtime logging and control through manual stop capabilities. Active agents appear in the "Active" tab, and their complete runtime logs are available in the "Past" tab after stopping.

- **Human-in-the-Loop Workflows**: Create sophisticated flows with nodes that wait for human response before continuing—enabling approval workflows, manual review processes, and collaborative AI-human decision making.

- **Perpetual Runtime**: Flows maintain state across containers, individual runtimes, and wait periods (no matter how long), enabling complex, long-running processes that persist through any interruption.

### Key Technical Advantages
- **Lightning-Fast Routing**: Recognition nodes complete round-trip decisions in ~1/3 second
- **Parallel Processing**: Child flows share state with parents and can run in parallel for exponential AI processing power
- **Persistent Runtime**: Flows maintain state across containers, individual runtimes, and wait periods (no 
matter how long)
- **Universal Integration**: Run flows programmatically in Python/JS or deploy as embeddable widgets

**Real-world example**: Aegis AI powers their entire "AI Firewall" security platform using Vector Flow as the core decision engine, processing thousands of real-time threat assessments daily. What started as a no-code prototype became their mission-critical production system—no rewrite required.

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

This dual-pane setup is your debugging superpower—you can see the AI's "thought process" and iterate until it's perfect. For detailed information about interpreting and using the logging system, see the [Vector Flow Logging documentation](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_logging.md).

## Your First Flow: The Dog Expert

Let's build a simple but powerful example—an AI agent that only talks about dogs. This demonstrates the core concepts of recognition, routing, and responses.

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

This simple flow demonstrates the power of Vector Flow—you've created an intelligent agent with conditional logic in just a few minutes, no coding required!

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
- **ACT**: Write Python code for API calls, data processing, or custom logic - [ACT Node Documentation](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_nodes/vf_act_node.md)
- **EMAIL**: Send automated emails with dynamic content
- **DOWNLOAD URL**: Fetch content from websites
- **GOOGLE SEARCH**: Perform searches and use results in your flow
- **MCP SERVER**: Connect to external tools and services - [MCP Node Documentation](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_nodes/vf_mcp_node.md)

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
- **ACT** sets variables via Python: `save['customer_data'] = api_response` - [ACT Node Documentation](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_nodes/vf_act_node.md)

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

### Performance Optimization
- Set low timeouts on RECOGNIZE nodes (1-2 seconds) for faster decisions
- Use "Get Context" sparingly—enable only when you need vault data for accuracy
- Leverage parallel processing for independent operations

## Next Steps: From Flow to Production

Once your flow works perfectly in the chat interface:

1. **Deploy as Widget**: Create embeddable chat widgets for websites
2. **Programmatic Integration**: Call flows from Python/JS applications with `vault.run_flow()` or `vault.stream_flow()` (see [Backend Runtime & Integration](https://github.com/John-Rood/VectorVault/tree/main/vectoflow_docs) and [JavaScript Integration](https://github.com/John-Rood/VectorVault-js/blob/main/vectorflow_docs.md) docs)
3. **Advanced Features**: Explore model selection, context retrieval, and temperature controls (see [Advanced Flow Builder Features](https://github.com/John-Rood/VectorVault/blob/main/vectoflow_docs/vectorflow_advanced_features.md))

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

For complete integration examples and API reference, see the [Backend Runtime & Integration documentation](https://github.com/John-Rood/VectorVault/tree/main/vectoflow_docs).

Vector Flow handles the orchestration—you focus on building intelligent agent behavior. Welcome to the Agent Operating System.

## Ready to Build?

Head to **vectorvault.io/vector-flow** and start with a simple Start → Recognize → Respond flow. Use the chat interface to test, watch the logs to debug, and iterate until your agent behaves exactly as intended. Then scale up to more complex patterns as you get comfortable with the platform.

Remember: Every sophisticated AI agent started with a simple flow. The key is iteration and using the real-time feedback from the chat/log interface to perfect your agent's behavior. 