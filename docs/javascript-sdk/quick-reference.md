# VectorVault JavaScript Client - Quick Reference

A quick reference guide for the VectorVault API. For detailed documentation, see `API_DOCUMENTATION.md`.

## Installation

```bash
npm install vectorvault
```

## Quick Start

```javascript
import VectorVault from 'vectorvault';

const vv = new VectorVault();
await vv.login('email@example.com', 'password');
// OR
await vv.initializeDeployment('email@example.com', 'deployment_id');
```

---

## Authentication

| Method | Signature | Description |
|--------|-----------|-------------|
| `login()` | `(email, password) => Promise<void>` | Login with credentials |
| `initializeDeployment()` | `(email, deploymentId) => Promise<void>` | Initialize deployment auth |
| `logout()` | `() => void` | Clear authentication |
| `getAccessToken()` | `() => string \| null` | Get current access token |
| `refreshAccessToken()` | `(maxRetries?) => Promise<boolean>` | Refresh token manually |

---

## Chat & RAG

### getChat()
```javascript
const response = await vv.getChat({
  vault: 'my_vault',
  text: 'Your question',
  get_context: true,
  n_context: 4,
  model: 'gpt-4o',
  temperature: 0.7
});
```

**Key Parameters:** `vault`, `text`, `get_context`, `n_context`, `return_context`, `model`, `temperature`, `history`, `custom_prompt`

### getChatStream()
```javascript
await vv.getChatStream(
  { vault: 'my_vault', text: 'Question', get_context: true },
  (chunk) => process.stdout.write(chunk)
);
```

---

## Vault Management

| Method | Signature | Description |
|--------|-----------|-------------|
| `getVaults()` | `() => Promise<string[]>` | List all vaults |
| `createVault()` | `(vault) => Promise<any>` | Create new vault |
| `deleteVault()` | `(vault) => Promise<any>` | Delete entire vault |
| `getAccountData()` | `() => Promise<any>` | Get account info |

```javascript
const vaults = await vv.getVaults();
await vv.createVault('new_vault');
```

---

## Data Management

### Adding Data

```javascript
// Add text
await vv.addCloud({
  vault: 'my_vault',
  text: 'Content to add',
  meta: { source: 'manual' },
  split: true,
  split_size: 1000
});

// Scrape website
await vv.addSite({
  vault: 'my_vault',
  site: 'https://example.com'
});

// Upload PDF (browser)
await vv.uploadPdf(fileObject, 'my_vault', {
  summarize: true,
  splitSize: 500
});
```

### Managing Items

```javascript
// Get items
const items = await vv.getItems('my_vault', [1, 2, 3]);

// Edit item
await vv.editItem('my_vault', 42, 'New text');

// Delete items
await vv.deleteItems('my_vault', [1, 2, 3]);

// Count items
const total = await vv.getTotalItems('my_vault');
```

### Import/Export

```javascript
// Export
const data = await vv.downloadToJson({
  vault: 'my_vault',
  return_meta: true
});

// Import
await vv.uploadFromJson('my_vault', jsonData);
```

---

## Search & Similarity

### getSimilar()

**Single Vault:**
```javascript
const results = await vv.getSimilar({
  vault: 'my_vault',
  text: 'search query',
  num_items: 5,
  include_distances: true
});
```

**Cross-Vault (Array):**
```javascript
const results = await vv.getSimilar({
  vaults: ['vault1', 'vault2', 'vault3'],
  text: 'search query',
  num_items: 10
});
```

**Cross-Vault (With Minimums):**
```javascript
const results = await vv.getSimilar({
  vaults: {
    technical_docs: 3,
    user_guides: 2,
    faqs: 1
  },
  text: 'search query',
  num_items: 10,
  include_distances: true
});
```

### getDistance()
```javascript
const result = await vv.getDistance('my_vault', id1, id2);
console.log(result.distance);
```

---

## Customization

```javascript
// Personality message
await vv.savePersonalityMessage('my_vault', 'You are a helpful assistant...');
const personality = await vv.fetchPersonalityMessage('my_vault');

// Custom prompt
await vv.saveCustomPrompt('my_vault', 'Answer: {context}\n\nQ: {question}');
const prompt = await vv.fetchCustomPrompt('my_vault');
```

---

## Flow Execution

> See `vectorflow_docs.md` for comprehensive flow documentation

### runFlow() - Get complete response
```javascript
const result = await vv.runFlow(
  'flow_name',
  'user message',
  '',                          // history (optional)
  'user_123',                  // conversation_user_id (optional)
  'session_abc',               // session_id (optional)
  'web',                       // invoke_method (optional)
  { custom_var: 'value' },     // internal_vars (optional)
  {                            // callbacks (optional)
    onMessage: (msg) => console.log(msg),
    onError: (err) => console.error(err)
  }
);

console.log(result.response);
console.log(result.logs);
```

### runFlowStream() - Stream response
```javascript
const result = await vv.runFlowStream(
  'flow_name',
  'user message',
  '',
  null,
  null,
  null,
  { var1: 'value' },
  {
    onLog: (log) => console.log('[LOG]', log),
    onMessage: (chunk) => process.stdout.write(chunk),
    onError: (err) => console.error(err)
  }
);
```

---

## Utility Methods

```javascript
// 3D map visualization data
const mapData = await vv.fetch3DMap('my_vault', highlightId);
```

---

## Common Patterns

### Error Handling
```javascript
try {
  await vv.getChat({ vault: 'my_vault', text: 'Hello' });
} catch (error) {
  if (error.message.includes('Session expired')) {
    await vv.login(email, password);
    // Retry operation
  }
  console.error('Error:', error.message);
}
```

### Streaming Chat to Console
```javascript
let fullResponse = '';
await vv.getChatStream(
  { vault: 'my_vault', text: 'Question', get_context: true },
  (chunk) => {
    fullResponse += chunk;
    process.stdout.write(chunk);
  }
);
console.log('\n\nComplete:', fullResponse);
```

### Building a Knowledge Base
```javascript
// Create vault
await vv.createVault('kb');

// Add documents
await vv.addCloud({ vault: 'kb', text: doc1, split: true });
await vv.addSite({ vault: 'kb', site: 'https://docs.example.com' });
await vv.uploadPdf(pdfFile, 'kb', { summarize: true });

// Query it
const response = await vv.getChat({
  vault: 'kb',
  text: 'What is the main topic?',
  get_context: true,
  n_context: 5
});
```

### Cross-Vault Search
```javascript
// Search multiple vaults, ensure minimums from each
const results = await vv.getSimilar({
  vaults: {
    'technical_docs': 2,
    'user_feedback': 1,
    'product_specs': 1
  },
  text: 'feature request',
  num_items: 10,
  include_distances: true
});

results.forEach(item => {
  console.log(`${item.vault}: ${item.text} (${item.distance})`);
});
```

---

## Parameter Quick Reference

### getChat() / getChatStream()
- `vault` - Vault name
- `text` - User message
- `get_context` - Retrieve context (boolean)
- `n_context` - Number of context items (number)
- `return_context` - Include context in response (boolean)
- `model` - LLM model (string, default: 'gpt-4o')
- `temperature` - Creativity (0-2, default: 0)
- `history` - Conversation history (string)
- `custom_prompt` - Custom system prompt (string)
- `timeout` - Request timeout in seconds (number)

### addCloud()
- `vault` - Target vault (required)
- `text` - Content to add (required)
- `meta` - Metadata object
- `name` - Item name/identifier
- `split` - Split into chunks (boolean)
- `split_size` - Chunk size (number)
- `gen_sum` - Generate summary (boolean)

### getSimilar()
- `text` - Search query (required)
- `vault` - Single vault name
- `vaults` - Cross-vault search (string | string[] | object)
- `num_items` - Max results (number, default: 4)
- `include_distances` - Include similarity scores (boolean)

---

## TypeScript Types

```typescript
interface FlowCallbacks {
  onMessage?: (message: string) => void;
  onLog?: (log: any) => void;
  onError?: (error: any) => void;
}

interface FlowResult {
  response: string;
  logs: any[];
}

interface PDFUploadOptions {
  summarize?: boolean;
  splitSize?: number;
}
```

---

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Login failed" | Invalid credentials | Check email/password |
| "Session expired" | Token expired | Call `login()` again |
| "Request failed: 404" | Endpoint not found | Verify method & parameters |
| "Request failed: 401" | Unauthorized | Re-authenticate |
| Network errors | Connection issue | Check network, retry |

---

## More Resources

- **Full API Documentation**: `API_DOCUMENTATION.md`
- **VectorFlow Guide**: `vectorflow_docs.md`
- **Logging Reference**: `vectorflow_logging.md`
- **GitHub**: https://github.com/John-Rood/vectorvault-js
- **Website**: https://vectorvault.io


