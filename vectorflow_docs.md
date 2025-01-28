# Vault Flow Methods Documentation

The Vault class provides methods for executing and streaming flows using a configured vault environment. These methods facilitate interaction with the Decision AI platform.

## Methods

### run_flow

Executes a flow synchronously and returns the complete response.

```python
def run_flow(
    self, 
    flow_name: str, 
    message: str, 
    history: str = '', 
    vault = None
) -> Any
```

#### Parameters:
- `flow_name` (str): The name of the flow to execute (e.g., 'some flow name')
- `message` (str): The input message to be processed by the flow
- `history` (str, optional): Previous conversation history. Defaults to empty string
- `vault` (Any, optional): Override the default vault configuration. If not provided, uses the instance's vault

#### Returns:
- Returns the flow execution result (type depends on the specific flow)

#### Example:
```python
vault = Vault(
    user='your@email.com',
    api_key='your-api-key',
    openai_key='your-openai-key',
    vault='vault name',
    verbose=False
)

response = vault.run_flow(
    flow_name='project tracker',
    message="What is the status of project X?",
    history="Previous conversation context..."
)
```

### stream_flow

Streams the flow execution results, yielding responses as they become available.

```python
def stream_flow(
    self, 
    flow_name: str, 
    message: str, 
    history: str = '', 
    vault = None
) -> Generator
```

#### Parameters:
- `flow_name` (str): The name of the flow to execute (e.g., 'dri artemis')
- `message` (str): The input message to be processed by the flow
- `history` (str, optional): Previous conversation history. Defaults to empty string
- `vault` (Any, optional): Override the default vault configuration. If not provided, uses the instance's vault

#### Returns:
- Generator that yields flow execution results as they become available

#### Example:
```python
vault = Vault(
    user='your@email.com',
    api_key='your-api-key',
    openai_key='your-openai-key',
    vault='vault name',
    verbose=False
)

for chunk in vault.stream_flow(
    flow_name='project tracker',
    message="What is the status of project X?",
    history="Previous conversation context..."
):
    print(chunk)  # Process streaming responses
```

## Implementation Details

Both methods internally use the instance's configured parameters:
- `self.user`: The authenticated user
- `self.api`: The API key for authentication
- `self.vault`: The default vault configuration
- `self.cuid`: The conversation user ID

The methods will use the instance's default vault unless explicitly overridden by passing a vault parameter.

## Common Use Cases

1. **Synchronous Processing**: Use `run_flow()` when you need the complete response before proceeding
2. **Real-time Updates**: Use `stream_flow()` for real-time processing of responses, useful for:
   - Progress updates
   - Long-running operations
   - Interactive chat interfaces

## Notes

- Both methods require proper initialization of the Vault instance with valid credentials
- The vault parameter allows for dynamic switching between different vault configurations
- Error handling should be implemented by the caller
- Stream processing may require appropriate handling of connection timeouts and interruptions