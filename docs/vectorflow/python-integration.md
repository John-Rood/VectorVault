# Vector Flow: The Execution Layer for Autonomous AI Agents

## Introduction: Beyond Chatbots

VectorFlow is the platform for building, deploying, and operating autonomous AI agents. While most tools focus on creating conversational AI, VectorFlow is engineered to solve the more complex challenge of running these systems reliably in production.

It provides the infrastructure to move beyond simple request-response bots and create **persistent, stateful agents** that execute complex tasks over time, with or without human supervision.

## A New Execution Layer for AI

### The Limits of Request-Response
Most AI today follows a stateless pattern: a user sends a message, the system returns an answer. This paradigm is great for chatbots but fails when building systems that must:

- Remember context across multiple, asynchronous interactions.
- Operate independently over hours or days.
- Handle complex, multi-step processes.
- Maintain state while scaling in a serverless environment.
- Provide a transparent, auditable log of their decision-making.

VectorFlow was built to solve these problems.

### The Persistent Agentic Runtime (PAR)
VectorFlow is powered by a **Persistent Agentic Runtime (PAR)**. Instead of rebuilding context with every interaction, agents maintain continuous state. Instead of waiting for human input, they operate with temporal autonomy, responding to events in real time.

This isn't a toolkit for building better chatbots. It is an **execution layer** for AI systems that need to work when no one is watching.

### Beyond Prompt Engineering
The market paradigm is still "ask → answer." VectorFlow operates on a different model:

- **Agents that persist and adapt:** Systems that learn and evolve over time.
- **Visual logic construction:** An interface for building complex reasoning, not just writing prompts.
- **Operational management:** A console for managing digital workers, not just chat widgets.

Predictable, observable, and maintainable is the difference.

---

## How It Works: From Visual Design to Autonomous Execution

### 1. Build Your Agent Visually
- Design and build flows at **[app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)**.
- Integrate leading models from OpenAI, Anthropic, xAI (Grok), and Google.
- Implement advanced reasoning patterns like Chain of Thought (COT), Tree of Thought (TOT), and Graph of Thought (GOT).

### Core Capabilities
- **API Integration:** Call external APIs within any flow.
- **Python Execution:** Run Python scripts in secure, sandboxed containers.
- **Variable System:** Use `{variable}` notation to pass data between nodes, inject runtime parameters, and access environment variables in your code.
- **Global Context:** All nodes have access to the current message and conversation history by default.

### 3. Deploy Instantly
- **Public Demo Pages:** Deploy shareable web demos directly from the UI.
- **Multi-Platform Integration:** Connect via JavaScript, Python, Zapier, or webhooks.
- **Real-time Updates:** Changes saved in the UI are reflected in production instantly.

### 4. Run on a Scalable Architecture
- **Serverless & Stateless:** Hosted on Google Cloud Platform (GCP) for reliability and scale.
- **VFJSON Format:** Flows are saved as a portable VFJSON object in the cloud.
- **Secure Execution:** Code runs in isolated containers with controlled environments and restricted module access.
- **Real-time Logging:** Get complete, auditable visibility into every execution step, output, and error. See [Vector Flow Logging](logging.md) for detailed logging documentation.

---
## Python API Documentation

### `run_flow()`

Executes a flow and returns the complete response. Best for synchronous operations where you need the final result before continuing.

```python
def run_flow(
    self, 
    flow_name: str, 
    message: str, 
    history: str = '', 
    invoke_method: str = None, 
    internal_vars: dict = None,
    **kwargs
) -> str
```

#### Parameters:
- **`flow_name`** (str): The name of the flow to execute
- **`message`** (str): The input message to start the flow
- **`history`** (str, optional): Previous conversation context
- **`invoke_method`** (str, optional): A string describing the invocation method for logging purposes (e.g., "web_app", "backend_script")
- **`internal_vars`** (dict, optional): A dictionary of runtime variables to inject into the flow.
- **`**kwargs`**: Any additional keyword arguments are also injected as runtime variables, providing a flexible way to pass parameters

#### Returns:
- **str**: Complete response from flow execution

#### Example:
```python
vault = Vault(
    user='your@email.com',
    api_key='your-api-key',
    openai_key='your-openai-key',
    vault='vault_name'
)

# Variables can be passed via internal_vars or directly as keyword arguments
response = vault.run_flow(
    flow_name='customer_support',
    message="I need help with my order",
    history="User: Hi\n\nBot: Hello!",
    invoke_method="support_dashboard",
    customer_id="12345",  # Passed via kwargs
    internal_vars={
        "customer_tier": "premium",
        "order_history": ["order_1", "order_2"]
    }
)
```

### `stream_flow()`

Executes a flow and streams the response in real-time. Best for interactive applications.

```python
def stream_flow(
    self, 
    flow_name: str, 
    message: str, 
    history: str = '', 
    invoke_method: str = None, 
    internal_vars: dict = None,
    **kwargs
) -> Generator
```

#### Parameters:
Same as `run_flow()` above.

#### Returns:
- **Generator**: Yields real-time events, including response chunks and logs, as they are produced by the flow.

#### Example:
```python
for event in vault.stream_flow(
    flow_name='data_processor',
    message="Analyze this dataset",
    dataset_id="dataset_123" # Passed via kwargs
):
    if event.startswith('data:'):
        # Process the streaming AI response
        print(event[5:], end='') # Remove 'data:' prefix
    elif event.startswith('event: log'):
        # Process structured log events for real-time observability
        # See Vector Flow Logging docs for log structure details
        continue
```

---

## Advanced Features & Patterns

### Dynamic Variable Injection with `kwargs`

The `**kwargs` parameter in the `run_flow` methods allows you to dynamically control a flow's behavior from your code without ever touching the UI. This is a powerful feature for turning generic agents into specialized workers on the fly.

#### Example: Dynamically Steering an AI Research Agent

Imagine a single, powerful research flow that can be steered to achieve different goals at runtime.

**In your Flow:**
- A `Generate` node creates a search query using the user's message and a `{research_goal}` variable.
- A `Google Search` node executes that query.
- Another `Generate` node synthesizes the results, also guided by the `{research_goal}`.
- A `Respond` node presents the final report.

**In your Python code:**
```python
# Steer the agent to act as a financial analyst
financial_report = vault.run_flow(
    flow_name='universal_researcher',
    message="What are the latest trends in renewable energy stocks?",
    research_goal="Provide a concise financial analysis for an investor."
)

# Steer the same agent to act as a technical writer
technical_summary = vault.run_flow(
    flow_name='universal_researcher',
    message="What are the latest trends in renewable energy stocks?",
    research_goal="Explain the underlying technology trends for a technical audience."
)
```
Here, the `research_goal` kwarg fundamentally changes the agent's behavior at each step, turning a generic tool into a specialized one.

### Variable Injection for AI Personalization
You can unpack a dictionary of user data to provide deep context, allowing the AI to generate truly personalized, empathetic responses.

```python
# Customer-specific variables available to AI
customer_data = {
    "name": "John Doe",
    "tier": "premium",
    "purchase_history": ["laptop", "mouse", "keyboard"],
    "support_tickets": 2,
    "last_purchase_date": "2024-01-15"
}

response = vault.run_flow(
    flow_name='intelligent_support',
    message="I need help with my recent order",
    **customer_data # Unpack dictionary as kwargs
)

# The AI has access to these variables and uses them intelligently:
# User: "I need help with my recent order"
# AI: "I see you purchased a laptop on January 15th, John. As one of our 
#      premium customers with a strong purchase history, I'd be happy to 
#      help. I notice this is only your third support request despite being 
#      with us for a while - what specific issue are you experiencing with 
#      your laptop?"
```

### Flow Inception - A Flow Within A Flow
Flows can run other flows as a part of their own execution. This allows you to build complex agents from smaller, reusable, single-purpose components.
```python
# A main router flow can call specialized flows for sales, support, etc.
# The parent and child flows automatically share state and context.
# No additional code needed - this is handled by the `Run Flow` node.
```

---

## Observability & Logging

Vector Flow provides comprehensive, structured JSON logging for complete traceability. For detailed information about log structure, debugging, and advanced logging features, see the [Vector Flow Logging](logging.md) documentation.

### Log Structure
Every log entry contains:
- **`node_id` & `node_name`**: The specific node that produced the log.
- **`run_id`**: A unique ID for a single node's execution instance, allowing you to group all events for one run.
- **`type`**: The event type (`node_start`, `script_output`, `node_complete`, etc.).
- **`start_time`, `end_time`, `processing_time`**: Precise performance metrics.
- **`params` or `error`**: Detailed context about the operation.

### Key Log Types
- **`node_start`**: Signals the beginning of a node's execution, with parameters.
- **`script_output`**: Captures any print statements from an `Act` node.
- **`script_error`**: Captures the full traceback if an `Act` node fails.
- **`node_complete`**: Signals the end of a node's execution, with results.

---

## Node Types Reference

### Core Logic
- **Start**: The designated entry point for every flow.
- **Recognize**: Answers a yes/no question about the user's intent to branch the logic.
- **Multiple Choice (list_match)**: AI-powered router that categorizes a message into one of several predefined paths.
- **If/Then**: Checks the value of a variable to branch the flow, enabling traditional conditional logic.

### AI & Response Generation
- **Respond**: The agent's voice. Delivers a static or AI-generated response to the user.
- **Generate**: An internal creative engine. Generates text and saves it to a variable for use in other nodes.
- **Number**: Extracts or calculates a numerical value and saves it to a variable.

### Data & State Management
- **Capture**: Turns conversational text into structured data by extracting specific details into variables.
- **Variable**: Sets or modifies a variable with a static value, essential for managing state.
- **Storage**: The agent's long-term memory. Saves and retrieves data from a persistent key-value store.
- **Add To Vault**: Makes the agent smarter by adding new information to your vector database.

### External Integrations
- **Act**: The gateway to the outside world. Runs Python code to call APIs, perform calculations, or execute custom logic.
- **Email**: Sends an email via any SMTP server, with dynamic content from variables.
- **Google Search**: Performs a real-time web search to give the agent access to current information.
- **Download URL (website)**: Lets the agent "read" a webpage by ingesting its content into a variable.

### Flow Control
- **Run Flow**: Enables modular design by allowing one flow to execute another as a sub-routine.
- **Parallel**: Speeds up execution by running multiple branches of a flow simultaneously.
- **Wait**: Pauses the flow for a specified duration, from seconds to days.
- **No Response**: Cleanly ends a flow without sending a final message to the user.

---

## Development & Deployment

### Best Practices for Building Agents
1.  **Start Simple**: Begin with linear flows and incrementally add branching and complexity.
2.  **Use Variables**: Leverage the variable system for dynamic, reusable logic.
3.  **Build Modularly**: Break large processes into smaller, single-purpose flows and connect them with the **Run Flow** node.
4.  **Optimize Performance**: Use **Parallel** nodes for independent, long-running tasks and select the right model size for the job.
5.  **Manage Security**: Use environment variables for secrets and validate all user inputs.

### Deployment Options
- **Public Demo Pages**: Instant, shareable web interfaces for your agents.
- **API & Webhooks**: Integrate with any application via a RESTful API.
- **Platform SDKs**: Use native clients for **JavaScript/React** and **Python**.
- **No-Code Integration**: Connect to thousands of apps with **Zapier**.

---

## Conclusion

VectorFlow provides the foundational infrastructure for developing, deploying, and managing the next generation of autonomous AI systems. It bridges the gap between conversational AI and production-grade agents, enabling you to build systems that are not only intelligent but also robust, scalable, and observable.

**Get started at [app.vectorvault.io/vector-flow](https://app.vectorvault.io/vector-flow)**
