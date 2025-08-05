# Advanced Features in Flow Builder: Documentation

Welcome to the advanced features documentation for Flow Builder! These features unlock powerful customizations for your AI flows, allowing fine-tuned control over AI models, data integration, variable handling, timeouts, and more. By clicking the \"Advanced\" toggle in any node's editor, you gain access to parameters that correspond to the underlying cloud function APIs (like `get_chat_stream()` for AI-powered nodes). This enables sophisticated behaviors such as context-aware AI responses, dynamic data injection, and optimized performance.

## High-Level Overview: What You Can Do with Advanced Features

At a high level, advanced features empower you to:
- **Customize AI Behavior**: Select specific models, adjust randomness (temperature), and control timeouts to optimize response speed and quality.
- **Integrate Data Sources**: Connect to vaults for vector-based context retrieval, enabling RAG (Retrieval-Augmented Generation) patterns where AI pulls relevant data automatically.
- **Manage Variables Dynamically**: Set, retrieve, and inject variables across nodes for data persistence and flow logic.
- **Handle Integrations and Actions**: Configure external services (e.g., email, MCP servers) with timeouts and output storage.
- **Optimize Performance**: Fine-tune context searches, smart history queries, and parallel execution for efficient flows.
- **Control Flow Logic**: Add conditions, waits, and branching with customizable parameters.

These features are particularly useful for building production-grade flows that handle real-world scenarios like personalized responses, data fetching via APIs, or conditional routing. Most advanced options map directly to parameters in backend functions like `get_chat_stream()`, ensuring seamless execution.

## Essential Knowledge: Bracket Variables and Variable Management

Before diving into node-specific features, the most important concept to understand is **bracket variables**—a powerful way to manage and inject data across your flow. 

- **Setting Variables**: Almost every node allows you to \"Set Variable\" in advanced settings, storing its output (e.g., AI response, extracted data, or computed value) in a named variable. Nodes like **CAPTURE** and **GENERATE** are specifically designed for this—CAPTURE uses AI to extract info (e.g., names, emails) from conversations (multiple at once), while GENERATE creates and sets a single variable with AI-generated text. Even action nodes like **RESPOND** can save the full streamed response to a variable for later use (e.g., in ACT or logs). The NUMBER node generates integers and sets them to variables for use in logic.
  
- **Injecting Variables with Bracket Notation**: Once set, you can reference variables anywhere using `{variable_name}`. For example, in a RESPOND node's prompt: \"Greet the user by their first name, {first_name}.\" The system automatically replaces the variable with the actual value at runtime. This works in any text field across nodes, enabling dynamic, data-driven flows.

- **Special Case: ACT Node Variables**: In Python scripts, variables are available in the global namespace—access them directly like `print(user_type)` if you've set `{user_type}` earlier. To pass data forward, assign to the `save` dictionary (e.g., `save['user_context'] = \"Customer details\"`). These become bracket variables for subsequent nodes, perfect for API fetches (e.g., pull customer data and inject it into a GENERATE node).

- **Why This Matters**: Bracket variables create seamless data flow without manual coding. They're especially useful for personalization (e.g., injecting API results) or chaining nodes (e.g., use a NUMBER output in an IF/THEN condition).

Pro Tip: Variables persist throughout the flow execution, but scope them logically to avoid conflicts. Use descriptive names like `{customer_id}` for clarity.

## Additional Things to Know

- **Timeouts for Optimization**: In RESPOND nodes, timeout measures \"time to first token\"—set it low (e.g., 1 second) to restart slow streams, preventing user delays. For RECOGNIZE nodes, a 1-2 second timeout speeds up yes/no decisions without sacrificing accuracy.
  
- **Default Variables**: Set global defaults (e.g., model or vault) in the flow's settings—these apply to all nodes unless overridden. For vaults, defaults won't trigger searches unless \"Get Context\" is enabled per node (off by default for speed; enable for auto-RAG use cases where vector search enhances AI accuracy).

- **Context and Search Nuances**: Features like \"Smart History Search\" let AI craft dynamic queries from conversation history, improving relevance in complex flows. Use sparingly for performance—combine with low n_context (e.g., 4 items) to balance speed and quality.

- **Best Practices**: Test variables in isolation (e.g., via ACT's print statements). For data-heavy flows, leverage vaults sparingly and prefer explicit API calls in ACT for precision.

Now, let's break it down by node type, focusing on advanced features and how they tie into backend parameters like those in `get_chat_stream()`.

## Advanced Features by Node Type

### AI-Powered Nodes

#### RESPOND Node
- **Model Selection**: Pick AI models (maps to `model`); supports custom finetuned models.
- **Vault Integration**: Specify vault for data retrieval (maps to vault params).
- **Get Context**: Enable vector search for relevant data (maps to `get_context`); default off for speed.
- **Context Count (n_context)**: Set items to retrieve (maps to `n_context`; default: 4).
- **Smart History Search**: AI generates search queries from history (maps to `smart_history_search`).
- **Temperature**: Adjust randomness (maps to `temperature`; 0-1, default: 0).
- **Timeout**: Set response timeout (maps to `timeout`; default: 45s; use low for first-token restarts).
- **Set Variable**: Store full response (after streaming) in a variable for later use (e.g., in ACT or logs).
- **Variable Name**: Name the variable (e.g., `{last_response}`).
- **Manual Response Toggle**: Switch to static text instead of AI.

#### RECOGNIZE Node
- **Model Selection**: Choose models, including finetuned (maps to `model`).
- **Temperature**: Control randomness (maps to `temperature`).
- **Timeout**: Optimize yes/no speed (maps to `timeout`; try 1-2s for performance).
- **Set Variable**: Store result (true/false) in a variable.
- **Variable Name**: Name for the result (e.g., `{is_valid}`).

#### GENERATE Node
- **Model Selection**: Select models (maps to `model`).
- **Vault Integration**: Choose vault.
- **Get Context**: Enable search (maps to `get_context`; default: on).
- **Context Count (n_context)**: Items to include (maps to `n_context`).
- **Smart History Search**: Dynamic queries (maps to `smart_history_search`).
- **Temperature**: Randomness (maps to `temperature`).
- **Timeout**: Response wait time (maps to `timeout`).
- **Variable Assignment**: Primary purpose is to generate text and set it to a single variable (similar to CAPTURE but for one item).

#### CAPTURE Node
- **Model Selection**: Choose models.
- **Temperature**: Randomness.
- **Timeout**: Extraction timeout.
- **Multiple Field Capture**: Dynamically add AI instructions/variables (e.g., extract `{first_name}`, `{email}`) and set multiple at once.

#### NUMBER Node
- **Model Selection**: Choose models.
- **Temperature**: Randomness.
- **Timeout**: Generation timeout.
- **Variable Assignment**: Store generated integer in a variable (e.g., for counts or scores).

#### MULTIPLE CHOICE (list_match) Node
- **Model Selection**: Choose models, including finetuned.
- **Temperature**: Randomness.
- **Timeout**: Choice selection timeout.
- **Set Variable**: Store selected option.
- **Variable Name**: Name the variable.
- **Dynamic Options**: Add unlimited choices.

### Action & Integration Nodes

#### ACT Node (Python Code Execution)
*For detailed ACT node documentation and examples, see: [ACT Node Documentation](https://github.com/John-Rood/VectorVault/blob/main/documentation/vectoflow_docs/vectorflow_nodes/vf_act_node.md)*

- **Code Editor**: Python-only with syntax highlighting.
- **Global Variables**: Access `{variables}` directly (e.g., `print(user_type)`).
- **Save Dictionary**: Set forward variables via `save['key'] = value` (e.g., for API data).
- **Environment Access**: Use USER, API_KEY, etc., for integrations.
- **Output/Error Views**: Expandable dropdowns for script results.

#### EMAIL Node
- **SMTP Setup**: Server, port, credentials.
- **Email Details**: From/to, subject, body (supports `{variables}`).

#### MCP SERVER Node
*For detailed MCP node documentation and examples, see: [MCP Node Documentation](https://github.com/John-Rood/VectorVault/blob/main/documentation/vectoflow_docs/vectorflow_nodes/vf_mcp_node.md)*

- **Server URL**: Custom endpoint.
- **Operations**: List/call tools, list/get prompts.
- **Arguments**: JSON for tools/prompts.
- **Timeout**: Server connection wait (default: 30s).
- **Set Variable**: Store response.
- **Variable Name**: For the response.

#### DOWNLOAD URL (website) Node
- **URL Input**: Target URL.
- **Set Variable**: Store content (default: on).
- **Variable Name**: For downloaded text.

#### GOOGLE SEARCH Node
- **Query Input**: Search terms (supports `{variables}`).
- **Variable Name**: Store results.

### Storage & Data Management Nodes

#### STORAGE Node
- **Actions**: Get/Set/Delete.
- **Path**: Storage location.
- **Value**: Data for Set.
- **Variable Name**: For Get results.
- **Vault**: Specific vault.

#### ADD TO VAULT Node
- **Data Input**: Content to add (supports `{variables}`).
- **Vault**: Target vault.

### Control Flow Nodes

#### IF/THEN Node
- **Conditions**: Multiple with operators (==, >, contains, etc.).
- **Logic**: AND/OR combining.
- **Variable Support**: Use `{variables}` in conditions; especially useful for variables from parallel operations (e.g., recognize multiple things at once, save to variables, then conditionally respond—only way for conditional logic on parallel results).

#### RUN FLOW Node
- **Flow Name**: Child flow to run.
- **Command**: Pass instructions.
- **Dead End**: Don't wait for response.
- **Set Variable**: Store child output.
- **Variable Name**: For the output.

### Simple Nodes

#### VARIABLE Node  
- **Static Value**: Set fixed data.
- **Variable Name**: Assign it.
