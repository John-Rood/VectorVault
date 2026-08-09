# VectorFlow Logging Reference

## Node Logging Structure

### RESPOND NODE
**Node Start:**
```json
{
  "node_id": "12345", "node_name": "Main Response", "run_id": "a1b2c3d4",
  "type": "node_start", "node_type": "respond", "start_time": 1703123456.789,
  "message": "Starting response generation with my_vault",
  "params": {"vault_name": "my_vault", "model": "gpt-4", "prompt": "...", "static_response": "", "get_context": true, "n_context": 4, "temperature": 0.0}
}
```

**Static Response (if static):**
```json
{
  "node_id": "12345", "node_name": "Main Response", "run_id": "a1b2c3d4",
  "type": "static_response", "message": "Using static response text", "response_length": 50
}
```

**First Token (if streaming):**
```json
{
  "node_id": "12345", "node_name": "Main Response", "run_id": "a1b2c3d4",
  "type": "first_token", "time_to_first_token": 1.234
}
```

**Node Complete:**
```json
{
  "node_id": "12345", "node_name": "Main Response", "run_id": "a1b2c3d4",
  "type": "node_complete", "end_time": 1703123459.234, "processing_time": 2.445,
  "response_length": 150, "was_static": false
}
```

### RECOGNIZE NODE
**Node Start:**
```json
{
  "node_id": "67890", "node_name": "Intent Check", "run_id": "e5f6g7h8",
  "type": "node_start", "node_type": "recognize", "start_time": 1703123456.789,
  "message": "Starting recognition with prompt: Is this a question about...",
  "params": {"vault_name": "my_vault", "model": "gpt-4", "prompt": "...", "timeout": 45}
}
```

**Retry (if needed):**
```json
{
  "node_id": "67890", "node_name": "Intent Check", "run_id": "e5f6g7h8",
  "type": "retry_attempt", "message": "Recognition failed, retrying: API timeout", "error": "API timeout"
}
```

**Node Complete:**
```json
{
  "node_id": "67890", "node_name": "Intent Check", "run_id": "e5f6g7h8",
  "type": "node_complete", "message": "Recognition result: YES",
  "end_time": 1703123458.234, "processing_time": 1.445, "recognition_result": true, "output_path": "output_1"
}
```

### GENERATE NODE
**Node Start:**
```json
{
  "node_id": "11111", "node_name": "Generate Summary", "run_id": "i9j0k1l2",
  "type": "node_start", "node_type": "generate", "start_time": 1703123456.789,
  "params": {"vault_name": "my_vault", "model": "gpt-4", "prompt": "...", "get_context": false, "n_context": 4, "smart_history_search": false, "temperature": 0.0, "timeout": 45, "var_name": "summary"}
}
```

**Node Complete:**
```json
{
  "node_id": "11111", "node_name": "Generate Summary", "run_id": "i9j0k1l2",
  "type": "node_complete", "message": "Generated 250 characters for 'summary'",
  "end_time": 1703123459.234, "response": "Generated content...", "processing_time": 2.445,
  "response_length": 250, "variable_set": "summary", "generated_content": "Generated summary content..."
}
```

### ACT NODE
**Node Start:**
```json
{
  "node_id": "22222", "node_name": "Process Data", "run_id": "m3n4o5p6",
  "type": "node_start", "node_type": "act", "start_time": 1703123456.789,
  "message": "Executing python script (245 characters)",
  "params": {"language": "python", "script": "print('Hello World!')", "current_vars": {"var1": "value1", "var2": "value2"}}
}
```

**Script Output (if script prints anything):**
```json
{
  "node_id": "22222", "node_name": "Process Data", "run_id": "m3n4o5p6",
  "type": "script_output", "message": "Script output: Hello World!",
  "output": "Hello World!"
}
```

**Script Error (if error occurred):**
```json
{
  "node_id": "22222", "node_name": "Process Data", "run_id": "m3n4o5p6",
  "type": "script_error", "message": "Error executing script: NameError: name 'x' is not defined",
  "error": "NameError: name 'x' is not defined"
}
```

**Node Complete:**
```json
{
  "node_id": "22222", "node_name": "Process Data", "run_id": "m3n4o5p6",
  "type": "node_complete", "end_time": 1703123457.123, "processing_time": 0.334,
  "execution_success": true, "script_output_length": 45, "vars_changed": true, "changed_vars_count": 3, "error": null
}
```

### NUMBER NODE
**Node Start:**
```json
{
  "node_id": "33333", "node_name": "Get Rating", "run_id": "q7r8s9t0",
  "type": "node_start", "node_type": "number", "start_time": 1703123456.789,
  "message": "Extracting number using prompt: What is the rating...",
  "params": {"vault_name": "my_vault", "model": "gpt-4", "prompt": "...", "timeout": 45}
}
```

**Node Complete:**
```json
{
  "node_id": "33333", "node_name": "Get Rating", "run_id": "q7r8s9t0",
  "type": "node_complete", "message": "Extracted number: 8.5",
  "end_time": 1703123458.123, "processing_time": 1.334, "number_response": 8.5
}
```

### VARIABLE NODE
**Node Start:**
```json
{
  "node_id": "44444", "node_name": "Set Status", "run_id": "u1v2w3x4",
  "type": "node_start", "node_type": "variable", "start_time": 1703123456.789,
  "message": "Setting variable 'status' to: processing",
  "params": {"value": "processing", "var_name": "status"}
}
```

**Node Complete:**
```json
{
  "node_id": "44444", "node_name": "Set Status", "run_id": "u1v2w3x4",
  "type": "node_complete", "message": "Variable 'status' set successfully",
  "end_time": 1703123456.792, "processing_time": 0.003, "variable_set": "status", "value_assigned": "processing"
}
```

### LIST_MATCH NODE
**Node Start:**
```json
{
  "node_id": "55555", "node_name": "Choose Category", "run_id": "y5z6a7b8",
  "type": "node_start", "node_type": "list_match", "start_time": 1703123456.789,
  "message": "Matching from 4 options",
  "params": {"vault_name": "my_vault", "model": "gpt-4", "prompt": "...", "options_count": 4, "options": ["Technical Support", "Billing", "Sales", "General"], "timeout": 45}
}
```

**Node Complete:**
```json
{
  "node_id": "55555", "node_name": "Choose Category", "run_id": "y5z6a7b8",
  "type": "node_complete", "message": "Selected option 2: 'Technical Support'",
  "end_time": 1703123458.234, "processing_time": 1.445, "selected_option": "Technical Support", "option_index": 2, "output_path": "output_2"
}
```

### CAPTURE NODE
**Node Start:**
```json
{
  "node_id": "66666", "node_name": "Extract Details", "run_id": "z9a8b7c6",
  "type": "node_start", "node_type": "capture", "start_time": 1703123456.789,
  "params": {"vault_name": "my_vault", "model": "gpt-4", "instructions_count": 3, "labels_count": 3, "timeout": 45, "parallel_execution": true}
}
```

**Node Complete:**
```json
{
  "node_id": "66666", "node_name": "Extract Details", "run_id": "z9a8b7c6",
  "type": "node_complete", "message": "Captured 3 variables in 2.34s",
  "end_time": 1703123459.123, "processing_time": 2.334, "variables_captured": 3, "execution_mode": "parallel"
}
```

### PARALLEL NODE
**Node Start:**
```json
{
  "node_id": "77777", "node_name": "Parallel Processor", "run_id": "d5e6f7g8",
  "type": "node_start", "node_type": "parallel", "start_time": 1703123456.789,
  "params": {"parallel_paths": 3, "next_node_id": "88888"}
}
```

**Node Complete:**
```json
{
  "node_id": "77777", "node_name": "Parallel Processor", "run_id": "d5e6f7g8",
  "type": "node_complete", "end_time": 1703123460.123, "processing_time": 3.334, "parallel_paths_completed": 3
}
```

### IF_THEN NODE
**Node Start:**
```json
{
  "node_id": "99999", "node_name": "Check Conditions", "run_id": "h9i0j1k2",
  "type": "node_start", "node_type": "if_then", "start_time": 1703123456.789,
  "params": {"conditions_count": 2, "conditions": [{"variable": "status", "operator": "==", "value": "active"}, {"variable": "score", "operator": ">", "value": "5"}]}
}
```

**Node Complete:**
```json
{
  "node_id": "99999", "node_name": "Check Conditions", "run_id": "h9i0j1k2",
  "type": "node_complete", "end_time": 1703123456.834, "processing_time": 0.045,
  "evaluation_result": true, "conditions_evaluated": 2, "output_path": "output_1"
}
```

### EMAIL NODE
**Node Start:**
```json
{
  "node_id": "11111", "node_name": "Send Notification", "run_id": "l3m4n5o6",
  "type": "node_start", "node_type": "email", "start_time": 1703123456.789,
  "message": "Sending email to user@example.com",
  "params": {"smtp_server": "smtp.gmail.com", "smtp_port": 587, "from_email": "system@company.com", "to_email": "user@example.com", "subject": "Notification", "body_length": 150}
}
```

**Node Complete:**
```json
{
  "node_id": "11111", "node_name": "Send Notification", "run_id": "l3m4n5o6",
  "type": "node_complete", "end_time": 1703123458.123, "processing_time": 1.334,
  "email_success": true, "to_email": "user@example.com", "subject": "Notification", "error": null
}
```

### RUN_FLOW NODE
**Node Start:**
```json
{
  "node_id": "22222", "node_name": "Call Helper Flow", "run_id": "p7q8r9s0",
  "type": "node_start", "node_type": "run_flow", "start_time": 1703123456.789,
  "message": "Running child flow: helper_flow",
  "params": {"flow_name": "helper_flow", "var_name": "helper_result", "command": null, "dead_end": false, "no_response_needed": false}
}
```

**Child Flow Log:**
```json
{
  "node_id": "22222", "node_name": "Call Helper Flow", "run_id": "p7q8r9s0",
  "type": "child_flow_log", "flow_name": "helper_flow", "child_log": "Processing started in helper flow"
}
```

**Node Complete:**
```json
{
  "node_id": "22222", "node_name": "Call Helper Flow", "run_id": "p7q8r9s0",
  "type": "node_complete", "end_time": 1703123465.234, "processing_time": 8.445,
  "flow_name": "helper_flow", "dead_end": false, "response_length": 145
}
```

### WAIT NODE
**Node Start:**
```json
{
  "node_id": "33333", "node_name": "Delay Process", "run_id": "t1u2v3w4",
  "type": "node_start", "node_type": "wait", "start_time": 1703123456.789,
  "message": "Waiting for 5 m (300s)",
  "params": {"wait_value": 5, "wait_unit": "m", "wait_seconds": 300, "is_short_wait": false}
}
```

**Node Complete (Short):**
```json
{
  "node_id": "33333", "node_name": "Delay Process", "run_id": "t1u2v3w4",
  "type": "node_complete", "message": "Wait completed (30s)",
  "end_time": 1703123486.834, "processing_time": 30.045, "wait_method": "time_sleep", "wait_seconds": 30
}
```

**Node Complete (Long):**
```json
{
  "node_id": "33333", "node_name": "Delay Process", "run_id": "t1u2v3w4",
  "type": "node_complete", "message": "Long wait scheduled (300s) via Cloud Tasks",
  "end_time": 1703123457.123, "processing_time": 0.334, "wait_method": "cloud_tasks", "wait_seconds": 300, "task_name": "task_abc123"
}
```

### WEBSITE NODE
**Node Start:**
```json
{
  "node_id": "44444", "node_name": "Fetch Content", "run_id": "x5y6z7a8",
  "type": "node_start", "node_type": "website", "start_time": 1703123456.789,
  "message": "Downloading content from https://example.com",
  "params": {"url": "https://example.com", "var_name": "page_content"}
}
```

**Node Complete:**
```json
{
  "node_id": "44444", "node_name": "Fetch Content", "run_id": "x5y6z7a8",
  "type": "node_complete", "end_time": 1703123458.234, "processing_time": 1.445,
  "url": "https://example.com", "download_success": true, "data_length": 12456, "variable_set": "page_content", "error": null
}
```

### STORAGE NODE
**Node Start:**
```json
{
  "node_id": "55555", "node_name": "Store Data", "run_id": "b9c0d1e2",
  "type": "node_start", "node_type": "storage", "start_time": 1703123456.789,
  "params": {"vault_name": "my_vault", "action": "set", "path": "/user/data", "value": "User data content..."}
}
```

**Node Complete:**
```json
{
  "node_id": "55555", "node_name": "Store Data", "run_id": "b9c0d1e2",
  "type": "node_complete", "end_time": 1703123457.123, "processing_time": 0.334,
  "action": "set", "path": "/user/data", "operation_success": true, "retrieved_value": null, "error": null
}
```

### ADD_VAULT NODE
**Node Start:**
```json
{
  "node_id": "66666", "node_name": "Add Document", "run_id": "f3g4h5i6",
  "type": "node_start", "node_type": "add_vault", "start_time": 1703123456.789,
  "message": "Adding 1543 characters to vault my_vault",
  "params": {"vault_name": "my_vault", "data_length": 1543, "data_preview": "This is the document content..."}
}
```

**Node Complete:**
```json
{
  "node_id": "66666", "node_name": "Add Document", "run_id": "f3g4h5i6",
  "type": "node_complete", "end_time": 1703123458.234, "processing_time": 1.445,
  "vault_name": "my_vault", "add_success": true, "data_length": 1543, "error": null
}
```

### GOOGLE_SEARCH NODE
**Node Start:**
```json
{
  "node_id": "77777", "node_name": "Web Search", "run_id": "j7k8l9m0",
  "type": "node_start", "node_type": "google_search", "start_time": 1703123456.789,
  "message": "Searching Google for: AI programming tutorials...",
  "params": {"search_input": "AI programming tutorials", "var_name": "search_results"}
}
```

**Node Complete:**
```json
{
  "node_id": "77777", "node_name": "Web Search", "run_id": "j7k8l9m0",
  "type": "node_complete", "end_time": 1703123458.234, "processing_time": 1.445,
  "search_input": "AI programming tutorials", "search_success": true, "results_count": 5, "variable_set": "search_results", "error": null
}
```

**Node Complete:**
```json
{
  "node_id": "88888", "node_name": "Start Node", "run_id": "n1o2p3q4",
  "type": "node_complete", "end_time": 1703123456.791, "processing_time": 0.002
}
```

### MCP NODE
*Model Context Protocol - Connect to external MCP servers to use tools and prompts*

**Operations Supported:**
- `list_tools` - Get available tools from MCP server
- `call_tool` - Execute a specific tool with arguments  
- `list_prompts` - Get available prompts from MCP server
- `get_prompt` - Retrieve a specific prompt with arguments

**Node Start:**
```json
{
  "node_id": "11111", "node_name": "Call External Tool", "run_id": "m1c2p3d4",
  "type": "node_start", "node_type": "mcp", "start_time": 1703123456.789,
  "message": "Starting MCP connection to http://localhost:3000",
  "params": {
    "server_url": "http://localhost:3000",
    "operation": "call_tool",
    "tool_name": "get_weather",
    "tool_arguments": "{\"location\": \"San Francisco\"}",
    "prompt_name": "",
    "prompt_arguments": "{}",
    "timeout": 30
  }
}
```

**Node Complete (Success):**
```json
{
  "node_id": "11111", "node_name": "Call External Tool", "run_id": "m1c2p3d4",
  "type": "node_complete", "message": "MCP operation 'call_tool' completed successfully",
  "end_time": 1703123458.234, "processing_time": 1.445,
  "operation": "call_tool", "response_length": 156
}
```

**Node Error (Failure):**
```json
{
  "node_id": "11111", "node_name": "Call External Tool", "run_id": "m1c2p3d4",
  "type": "node_error", "message": "MCP operation failed: Network error connecting to MCP server",
  "end_time": 1703123458.234, "processing_time": 1.445,
  "error": "Network error connecting to MCP server: Connection refused"
}
```

### NORESPONSE NODE
**Node Complete:**
```json
{
  "node_id": "99999", "node_name": "No Response", "run_id": "r5s6t7u8",
  "message": "No Response", "type": "node_complete", "parent_resumed": false
}
```

## System Logs
**Variable Assignment:**
```json
{"message": "VAR <> variable_name: variable_value"}
```

**Vault Operations:**
```json
{"message": "Vault Change: old_vault → new_vault"}
{"message": "Model Platform Change: gpt-3.5-turbo → gpt-4"}
{"message": "get vault time: 0.15 seconds"}
```

**Flow Processing:**
```json
{"message": "Processing Flow Name: my_flow"}
{"message": "**Node: ID=12345, Type=respond"}
```
