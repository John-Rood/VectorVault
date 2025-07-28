# Act Node Documentation

## Overview

The **Act node** is your gateway to custom logic and external integrations within Vector Flow. It executes Python code in a secure, sandboxed environment running on Google Cloud Platform containers, giving you the power to call APIs, perform complex calculations, manipulate data, and integrate with external systems while maintaining security and observability.

## Design Philosophy

**Act nodes are for integration, not implementation.** They're designed to:
- ✅ Call external APIs and services
- ✅ Process and transform data between nodes
- ✅ Implement conditional logic and calculations
- ✅ Extract and structure information
- ✅ Handle authentication and API workflows

**Act nodes are NOT for:**
- ❌ Building complex applications or codebases
- ❌ Installing external Python packages
- ❌ File system operations or data persistence
- ❌ Long-running background processes
- ❌ Heavy computational workloads

Think of Act nodes as the code that connects your flow to the outside world, not as a place to build your entire application logic.

## Execution Environment

### Containerized Runtime
- **Platform**: Google Cloud Platform (GCP) serverless containers
- **Language**: Python (fixed, no other languages supported)
- **Isolation**: Secure sandbox with restricted module access
- **Performance**: Real-time execution with full logging and error capture

### Resource Limits
- **Execution Timeout**: 10 minutes maximum
- **Memory Limit**: 1 GB RAM
- **CPU**: 2 cores available
- **Failure Handling**: Resource limit violations will generate traceback errors in logs

### Security Model
The Act node runs in a restricted environment designed for safety:

**Blocked Built-ins**: 
- File system access: `open`, `input`, `execfile`
- Dynamic execution: `exec`, `eval`, `compile`  
- Introspection: `globals`, `locals`, `vars`, `dir`
- System access: `help`, `exit`, `quit`

**Allowed Modules** (complete list):
```python
# Available for import - NO additional packages can be installed
import math
import datetime
import json
import random
import re
import collections
import itertools
import functools
import operator
import string
import decimal
import requests
```

**Important Limitation**: You cannot install additional Python packages or import external libraries. Act nodes are designed for API calls and data processing, not for building complex codebases. For advanced functionality, call external APIs that provide the services you need.

**Safe Built-ins**: Custom implementations of `getattr`, `setattr`, `delattr`, and `property` are provided for object manipulation.

## Variable System

### Accessing Flow Variables

All variables from your flow are automatically injected into the global scope:

```python
# If your flow has variables: customer_id, order_total, user_tier
print(f"Processing order for customer {customer_id}")
print(f"Order total: ${order_total}")
print(f"User tier: {user_tier}")
```

### Built-in Global Variables

**MESSAGE**: The current user input
```python
user_input = MESSAGE
print(f"User said: {user_input}")
```

**HISTORY**: Complete conversation history
```python
conversation = HISTORY
# Process previous conversation context
if "refund" in conversation.lower():
    print("User mentioned refunds previously")
```

### Credentials Access
Your VectorVault credentials are automatically available as global variables:
```python
# Access your credentials directly (no os.environ needed)
user_email = USER
api_key = API_KEY
openai_key = OPENAI_KEY
vault_name = VAULT

# You can create a new vault instance within your script
from vectorvault import Vault
vault = Vault(user=user_email, api_key=api_key, openai_key=openai_key, vault=vault_name)
```

### Saving Variables Back to Flow

Use the special `save` dictionary to persist variables for use in subsequent nodes:

```python
# Example: Process customer data and save results
customer_data = {
    "id": customer_id,
    "verified": True,
    "last_contact": datetime.datetime.now().isoformat()
}

# Save individual variables
save['customer_verified'] = True
save['verification_time'] = datetime.datetime.now().isoformat()
save['customer_data'] = customer_data

# These variables are now available to all subsequent nodes in your flow
```

**Supported Data Types**: Only JSON-serializable types can be saved and passed between nodes:
- `str`, `int`, `float`, `bool`
- `dict`, `list`
- Complex Python objects (classes, functions) will not transpose between nodes

## Practical Examples

### API Integration Example
```python
import requests
import json

# Call external API using flow variables
api_url = f"https://api.example.com/customers/{customer_id}"
headers = {"Authorization": f"Bearer {api_token}"}

try:
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    
    customer_info = response.json()
    
    # Save results for use in other nodes
    save['customer_name'] = customer_info.get('name')
    save['customer_tier'] = customer_info.get('tier', 'standard')
    save['api_success'] = True
    
    print(f"Successfully retrieved data for {customer_info.get('name')}")
    
except requests.exceptions.RequestException as e:
    save['api_success'] = False
    save['error_message'] = str(e)
    print(f"API call failed: {e}")
```

### Data Processing Example
```python
import json
import re

# Process and clean user input
cleaned_message = re.sub(r'[^\w\s]', '', MESSAGE.lower())
words = cleaned_message.split()

# Extract key information
order_numbers = [word for word in words if word.startswith('ord') and word[3:].isdigit()]
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, MESSAGE)

# Save extracted data
save['extracted_order_numbers'] = order_numbers
save['extracted_emails'] = emails
save['word_count'] = len(words)
save['processed_message'] = cleaned_message

print(f"Extracted {len(order_numbers)} order numbers and {len(emails)} emails")
```


## Logging & Observability

### Automatic Logging
Every Act node execution generates comprehensive JSON logs:

**Node Start**: Execution begins
```json
{
  "node_id": "12345",
  "node_name": "Process Customer Data",
  "run_id": "a1b2c3d4",
  "type": "node_start",
  "node_type": "act",
  "start_time": 1703123456.789,
  "message": "Executing python script (245 characters)",
  "params": {
    "language": "python",
    "script": "# Your code here...",
    "current_vars": {"customer_id": "12345", "order_total": 99.99}
  }
}
```

**Script Output**: All print statements
```json
{
  "node_id": "12345",
  "run_id": "a1b2c3d4", 
  "type": "script_output",
  "message": "Script output: Successfully processed customer data",
  "output": "Successfully processed customer data"
}
```

**Errors**: Full tracebacks for debugging
```json
{
  "node_id": "12345",
  "run_id": "a1b2c3d4",
  "type": "script_error", 
  "message": "Error executing script: NameError: name 'undefined_var' is not defined",
  "error": "Traceback (most recent call last):\n  File \"<string>\", line 3, in <module>\nNameError: name 'undefined_var' is not defined"
}
```

**Node Complete**: Execution summary
```json
{
  "node_id": "12345",
  "run_id": "a1b2c3d4",
  "type": "node_complete",
  "end_time": 1703123459.123,
  "processing_time": 2.334,
  "execution_success": true,
  "script_output_length": 45,
  "vars_changed": true,
  "changed_vars_count": 3,
  "error": null
}
```

### Debugging Best Practices

**Use Print Statements**: All output is captured and logged
```python
print(f"Debug: customer_id = {customer_id}")
print(f"Debug: API response status = {response.status_code}")
```

**Handle Exceptions Gracefully**: If an Act node throws an unhandled exception, any output before the error will still be logged, then the error traceback will be captured.
```python
try:
    # Your risky operation
    result = risky_api_call()
    save['success'] = True
    print("Operation completed successfully")
except Exception as e:
    print(f"Operation failed: {e}")
    save['success'] = False
    save['error_details'] = str(e)
    # Flow can continue - implement your own recovery logic
```

**Validate Inputs**:
```python
# Check if required variables exist
if 'customer_id' not in globals() or not customer_id:
    print("Error: customer_id is required but not provided")
    save['validation_error'] = "Missing customer_id"
    # Handle gracefully
else:
    # Proceed with normal logic
    print(f"Processing customer: {customer_id}")
```

## Common Patterns

### Conditional Logic
```python
# Use Act nodes for complex conditional logic
if customer_tier == "premium" and order_total > 1000:
    save['apply_discount'] = True
    save['discount_rate'] = 0.15
    print("Applied premium discount")
elif customer_tier == "gold":
    save['apply_discount'] = True  
    save['discount_rate'] = 0.10
    print("Applied gold discount")
else:
    save['apply_discount'] = False
    save['discount_rate'] = 0.0
    print("No discount applied")
```

### Data Transformation
```python
import json

# Transform data for downstream consumption
raw_data = api_response_data
transformed_data = {
    "customer": {
        "id": raw_data.get("customer_id"),
        "name": raw_data.get("full_name", "").title(),
        "email": raw_data.get("email_address", "").lower()
    },
    "preferences": {
        "notifications": raw_data.get("wants_notifications", True),
        "marketing": raw_data.get("marketing_opt_in", False)
    }
}

save['clean_customer_data'] = json.dumps(transformed_data)
print(f"Transformed data for customer {transformed_data['customer']['name']}")
```

### Error Recovery
```python
# Implement retry logic and fallbacks
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        save['api_data'] = response.json()
        save['api_success'] = True
        print(f"API call succeeded on attempt {retry_count + 1}")
        break
    except Exception as e:
        retry_count += 1
        print(f"Attempt {retry_count} failed: {e}")
        if retry_count >= max_retries:
            save['api_success'] = False
            save['final_error'] = str(e)
            print("All retry attempts failed, using fallback data")
            save['api_data'] = {"status": "offline", "message": "Service unavailable"}
```

## Performance Considerations

- **Execution Timeout**: 10-minute hard limit - plan accordingly for long-running operations
- **Memory Management**: 1 GB RAM limit - avoid loading large datasets into memory
- **CPU Usage**: 2 cores available - suitable for moderate computational tasks
- **API Calls**: Always implement timeouts and error handling for external calls
- **Variable Size**: Large variables in `save` affect flow performance - consider data optimization
- **Variable Timing**: Variables are injected immediately on node execution and changes are instantly available to subsequent nodes

## Integration with Other Nodes

### Passing Data Forward
```python
# Set up data for subsequent Generate or Respond nodes
save['user_context'] = f"Customer {customer_name} (ID: {customer_id}) is a {customer_tier} member"
save['next_action'] = "send_confirmation_email"
save['email_recipient'] = customer_email
```

### Using Data from Previous Nodes
```python
# Access data set by previous Capture or Generate nodes
if 'extracted_intent' in globals():
    print(f"Processing intent: {extracted_intent}")
    # Handle based on intent
else:
    print("No intent data available, using default handling")
```
