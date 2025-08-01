# MCP (Model Context Protocol) Node Documentation

## Overview
The MCP node enables your workflow to connect to remote MCP servers and execute various operations including listing available tools, calling tools, listing prompts, and retrieving prompts.

## Features
- **Dual Output Paths**: Success (output_1) and Error (output_2) paths for robust error handling
- **Multiple Operations**: Supports all core MCP operations
- **JSON-RPC Protocol**: Full compliance with MCP specification
- **Timeout Control**: Configurable connection timeouts
- **Variable Setting**: Store MCP responses in workflow variables
- **Advanced Error Handling**: Detailed error reporting and logging

## Supported Operations

### 1. List Tools
- **Purpose**: Retrieve all available tools from the MCP server
- **Required Fields**: Server URL only
- **Response**: Array of tool definitions with names, descriptions, and schemas

### 2. Call Tool
- **Purpose**: Execute a specific tool on the MCP server  
- **Required Fields**: Server URL, Tool Name
- **Optional Fields**: Tool Arguments (JSON)
- **Response**: Tool execution result

### 3. List Prompts
- **Purpose**: Retrieve all available prompts from the MCP server
- **Required Fields**: Server URL only
- **Response**: Array of prompt definitions with names and descriptions

### 4. Get Prompt
- **Purpose**: Retrieve a specific prompt from the MCP server
- **Required Fields**: Server URL, Prompt Name
- **Optional Fields**: Prompt Arguments (JSON)
- **Response**: Prompt content with any dynamic substitutions

## Configuration Fields

### Basic Settings
- **Server URL**: The endpoint URL of the MCP server (e.g., `https://mcp-server.example.com/api`)
- **Operation**: Select from list_tools, call_tool, list_prompts, or get_prompt

### Tool-Specific Fields (shown when operation = "call_tool")
- **Tool Name**: Name of the tool to execute
- **Tool Arguments**: JSON object with tool parameters

### Prompt-Specific Fields (shown when operation = "get_prompt")
- **Prompt Name**: Name of the prompt to retrieve
- **Prompt Arguments**: JSON object with prompt variables

### Advanced Settings
- **Timeout**: Connection timeout in seconds (default: 30)
- **Set Variable**: Option to store the response in a workflow variable
- **Variable Name**: Name of the variable to set (if enabled)

## Usage Examples

### Example 1: List Available Tools
```
Server URL: https://my-mcp-server.com/api
Operation: list_tools
```
This will return an array of all available tools on the server.

### Example 2: Call a Web Search Tool
```
Server URL: https://search-mcp-server.com/api  
Operation: call_tool
Tool Name: web_search
Tool Arguments: {"query": "latest AI news", "limit": 5}
```

### Example 3: Get a Dynamic Prompt
```
Server URL: https://prompt-server.com/api
Operation: get_prompt  
Prompt Name: email_template
Prompt Arguments: {"recipient": "John", "subject": "Meeting"}
```

## Error Handling

The MCP node provides comprehensive error handling:

### Network Errors
- Connection timeouts
- DNS resolution failures  
- HTTP status errors

### Protocol Errors
- Invalid JSON responses
- Missing JSON-RPC fields
- Server-reported errors

### Validation Errors
- Missing required fields
- Invalid JSON in arguments
- Malformed server URLs

## Output Paths

### Success Path (output_1)
- Triggered when MCP operation completes successfully
- Response data is available in the node's response variable
- Continues normal workflow execution

### Error Path (output_2)  
- Triggered when any error occurs during MCP operation
- Error message is set as the node's response
- Allows for error-specific handling in workflow

## JSON-RPC Protocol Details

The MCP node uses standard JSON-RPC 2.0 protocol:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {...}
  }
}
```

## Response Formats

### Tool List Response
```json
{
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web for information",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "limit": {"type": "number"}
        }
      }
    }
  ]
}
```

### Tool Call Response
```json
{
  "content": [
    {
      "type": "text", 
      "text": "Search results: Found 5 articles about AI..."
    }
  ],
  "isError": false
}
```

### Prompt Response
```json
{
  "description": "Email template generator",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Write an email to John about the meeting..."
      }
    }
  ]
}
```

## Variable Integration

### Storing MCP Responses
When "Set Variable" is enabled, the entire MCP response is stored in your specified variable:

```python
# In subsequent Act nodes, you can access:
mcp_result = mcp_response  # Your variable name
tools = mcp_result.get('tools', [])
for tool in tools:
    print(f"Available tool: {tool['name']}")
```

### Using Variables in MCP Arguments
You can reference flow variables in MCP arguments:

```json
{
  "search_query": "{user_input}",
  "user_id": "{customer_id}",
  "limit": 10
}
```

## Security Considerations

- **HTTPS Recommended**: Always use HTTPS URLs for production MCP servers
- **Authentication**: Include API keys in request headers or URL parameters as required by your MCP server
- **Input Validation**: All JSON arguments are validated before transmission
- **Timeout Protection**: Configurable timeouts prevent hanging connections
- **Error Isolation**: Failed MCP calls don't crash the entire flow - they route to error output

## Real-World Workflow Examples

### Example 1: Dynamic Tool Discovery and Execution
```
1. MCP Node (list_tools) → Store available tools
2. Capture Node → Ask AI to select best tool for user request  
3. MCP Node (call_tool) → Execute the selected tool
4. Respond Node → Present formatted results
```

### Example 2: Multi-Step Research Agent
```
1. MCP Node (call_tool: web_search) → Search for information
2. Capture Node → Process and filter search results
3. MCP Node (call_tool: summarize) → Summarize findings
4. Generate Node → Create final research report
```

### Example 3: Error-Resilient External Integration
```
1. MCP Node (call_tool) → Try primary service
   ├─ Success → Continue normal flow
   └─ Error → Fallback Node → Try alternative service
```

## Integration Tips

1. **Chain MCP Calls**: Use the success output to trigger additional MCP operations
2. **Error Recovery**: Route error output to fallback nodes or notification systems  
3. **Response Processing**: Use subsequent nodes to parse and act on MCP responses
4. **Variable Storage**: Enable variable setting to use MCP results in later workflow steps
5. **Tool Discovery**: Use list_tools before call_tool to dynamically choose available tools
6. **Prompt Templates**: Combine get_prompt with Generate nodes for dynamic prompt construction

## Troubleshooting

### Common Issues

**Connection Refused**
- Verify server URL is correct and accessible
- Check if server is running and accepting connections

**Invalid JSON Arguments**  
- Validate JSON syntax in tool/prompt arguments
- Use online JSON validators for complex objects

**Timeout Errors**
- Increase timeout value for slow MCP servers
- Check network connectivity and server performance

**Authentication Errors**
- Ensure proper API keys or tokens are included in server URL
- Verify server authentication requirements

### Debug Tips

1. Start with "list_tools" operation to verify connectivity
2. Check workflow logs for detailed error messages
3. Test MCP server manually with curl or Postman first
4. Use smaller timeout values during development for faster feedback