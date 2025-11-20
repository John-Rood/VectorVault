# VectorVault JavaScript Client - API Documentation

Version: 1.6.5

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Class: VectorVault](#class-vectorvault)
   - [Constructor](#constructor)
   - [Properties](#properties)
4. [Authentication Methods](#authentication-methods)
5. [Chat & RAG Methods](#chat--rag-methods)
6. [Vault Management Methods](#vault-management-methods)
7. [Data Management Methods](#data-management-methods)
8. [Search & Similarity Methods](#search--similarity-methods)
9. [Customization Methods](#customization-methods)
10. [Flow Execution Methods](#flow-execution-methods)
11. [File Upload Methods](#file-upload-methods)
12. [Utility Methods](#utility-methods)
13. [Type Definitions](#type-definitions)
14. [Error Handling](#error-handling)

---

## Introduction

VectorVault is a JavaScript client library that provides seamless integration with VectorVault's Cloud Vector Database. It enables developers to build advanced RAG (Retrieve and Generate) applications with minimal effort.

**Key Features:**
- Secure authentication and token management
- Vector database operations (CRUD)
- AI-powered chat with context retrieval
- Streaming responses for real-time applications
- Multi-vault support and cross-vault search
- Flow execution for complex AI workflows
- PDF document processing

---

## Installation

### NPM (Recommended for React/Node.js projects)

```bash
npm install vectorvault
```

### CDN (For browser-based HTML projects)

```html
<script src="https://cdn.jsdelivr.net/gh/John-Rood/VectorVault-js@main/dist/vectorvault.bundle.js"></script>
```

---

## Class: VectorVault

### Constructor

```typescript
new VectorVault(embeddingsModel?: string | null)
```

Creates a new instance of the VectorVault client.

**Parameters:**
- `embeddingsModel` (optional): String specifying the embeddings model to use. Defaults to `null`.

**Example:**
```javascript
import VectorVault from 'vectorvault';

const vv = new VectorVault();
// or with a specific embeddings model
const vvWithModel = new VectorVault('text-embedding-ada-002');
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `embeddingsModel` | `string \| null` | The embeddings model being used |
| `accessToken` | `string \| null` | Current JWT access token |
| `refreshToken` | `string \| null` | JWT refresh token |
| `tokenExpiresAt` | `number \| null` | Token expiration timestamp (milliseconds) |
| `baseUrl` | `string` | API base URL (`https://api.vectorvault.io`) |
| `vectorUrl` | `string` | Vector operations URL (`https://vectors.vectorvault.io`) |
| `deploymentId` | `string \| null` | Current deployment ID (if using deployment auth) |

---

## Authentication Methods

### login()

```typescript
async login(email: string, password: string): Promise<void>
```

Authenticates a user with email and password, obtaining JWT tokens.

**Parameters:**
- `email`: User's email address
- `password`: User's password

**Returns:** Promise that resolves when login is successful

**Throws:** Error if login fails

**Example:**
```javascript
await vv.login('user@example.com', 'password123');
```

---

### initializeDeployment()

```typescript
async initializeDeployment(email: string, deploymentId: string): Promise<void>
```

Initializes a deployment-based authentication session (token-based authentication alternative to password login).

**Parameters:**
- `email`: User's email address
- `deploymentId`: Deployment identifier

**Returns:** Promise that resolves when initialization is successful

**Throws:** Error if deployment initialization fails

**Example:**
```javascript
await vv.initializeDeployment('user@example.com', 'deploy_abc123');
```

---

### refreshAccessToken()

```typescript
async refreshAccessToken(maxRetries?: number): Promise<boolean>
```

Refreshes the access token using the refresh token. Called automatically by authenticated requests when token is near expiration.

**Parameters:**
- `maxRetries` (optional): Maximum number of retry attempts. Defaults to `3`.

**Returns:** Promise resolving to `true` if refresh successful, `false` otherwise

**Example:**
```javascript
const refreshed = await vv.refreshAccessToken();
if (!refreshed) {
  console.log('Token refresh failed, please log in again');
}
```

---

### getAccessToken()

```typescript
getAccessToken(): string | null
```

Returns the current access token.

**Returns:** Current access token or `null` if not authenticated

**Example:**
```javascript
const token = vv.getAccessToken();
```

---

### getRefreshToken()

```typescript
getRefreshToken(): string | null
```

Returns the current refresh token.

**Returns:** Current refresh token or `null` if not authenticated

---

### setAccessToken()

```typescript
setAccessToken(token: string): void
```

Sets the access token and updates the expiration time.

**Parameters:**
- `token`: JWT access token

**Example:**
```javascript
vv.setAccessToken('eyJhbGc...');
```

---

### setRefreshToken()

```typescript
setRefreshToken(token: string): void
```

Sets the refresh token.

**Parameters:**
- `token`: JWT refresh token

---

### logout()

```typescript
logout(): void
```

Clears all authentication tokens and logs out the user.

**Example:**
```javascript
vv.logout();
```

---

## Chat & RAG Methods

### getChat()

```typescript
async getChat(params: GetChatParams): Promise<ChatResponse>
```

Gets a chat response with optional context retrieval from vault.

**Parameters:**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `vault` | `string` | `''` | No | Vault name to query |
| `text` | `string` | `''` | No | User's input message |
| `embeddings_model` | `string` | instance default | No | Embeddings model to use |
| `history` | `string \| null` | `null` | No | Conversation history |
| `summary` | `boolean` | `false` | No | Whether to summarize response |
| `get_context` | `boolean` | `false` | No | Whether to retrieve context from vault |
| `n_context` | `number` | `4` | No | Number of context items to retrieve |
| `return_context` | `boolean` | `false` | No | Whether to include context in response |
| `smart_history_search` | `boolean` | `false` | No | Enable intelligent history search |
| `model` | `string` | `'gpt-4o'` | No | LLM model to use |
| `include_context_meta` | `boolean` | `false` | No | Include metadata with context |
| `custom_prompt` | `string \| boolean` | `false` | No | Custom system prompt |
| `temperature` | `number` | `0` | No | Model temperature (0-2) |
| `timeout` | `number` | `45` | No | Request timeout in seconds |

**Returns:** Promise resolving to chat response object

**Example:**
```javascript
const response = await vv.getChat({
  vault: 'my_knowledge_base',
  text: 'What is machine learning?',
  get_context: true,
  n_context: 3,
  return_context: true,
  model: 'gpt-4o',
  temperature: 0.7
});

console.log(response);
```

---

### getChatStream()

```typescript
async getChatStream(
  params: GetChatStreamParams,
  callback: (chunk: string) => void
): Promise<void>
```

Gets a streaming chat response, calling the callback for each chunk of data.

**Parameters:**

Same as `getChat()`, plus:
- `callback`: Function called with each streamed text chunk

Additional parameters supported:
- `metatag`: `string[]` - Array of metatags for filtering
- `metatag_prefixes`: `string[]` - Array of metatag prefixes
- `metatag_suffixes`: `string[]` - Array of metatag suffixes

**Returns:** Promise that resolves when streaming completes

**Example:**
```javascript
let fullResponse = '';

await vv.getChatStream(
  {
    vault: 'my_vault',
    text: 'Tell me about quantum computing',
    get_context: true,
    model: 'gpt-4o'
  },
  (chunk) => {
    fullResponse += chunk;
    process.stdout.write(chunk); // Stream to console
  }
);

console.log('\nComplete response:', fullResponse);
```

---

## Vault Management Methods

### getVaults()

```typescript
async getVaults(): Promise<string[]>
```

Retrieves list of all vaults for the authenticated user.

**Returns:** Promise resolving to array of vault names

**Example:**
```javascript
const vaults = await vv.getVaults();
console.log('Available vaults:', vaults);
// Output: ['vault1', 'vault2', 'knowledge_base']
```

---

### createVault()

```typescript
async createVault(vault: string): Promise<any>
```

Creates a new vault.

**Parameters:**
- `vault`: Name of the vault to create

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.createVault('my_new_vault');
```

---

### deleteVault()

```typescript
async deleteVault(vault: string): Promise<any>
```

Deletes an entire vault and all its contents. **This action is irreversible.**

**Parameters:**
- `vault`: Name of the vault to delete

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.deleteVault('old_vault');
```

---

### getAccountData()

```typescript
async getAccountData(): Promise<AccountData>
```

Retrieves account information including vault statistics.

**Returns:** Promise resolving to account data object containing vault information

**Example:**
```javascript
const accountData = await vv.getAccountData();
console.log(accountData);
```

---

## Data Management Methods

### addCloud()

```typescript
async addCloud(params: AddCloudParams): Promise<any>
```

Adds text data to a vault with optional processing.

**Parameters:**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `vault` | `string` | `''` | Yes | Target vault name |
| `text` | `string` | `''` | Yes | Text content to add |
| `embeddings_model` | `string` | instance default | No | Embeddings model |
| `meta` | `object \| null` | `null` | No | Metadata to attach |
| `name` | `string \| null` | `null` | No | Name/identifier for the item |
| `split` | `boolean` | `false` | No | Whether to split into chunks |
| `split_size` | `number` | `1000` | No | Chunk size if splitting |
| `gen_sum` | `boolean` | `false` | No | Generate summary |

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.addCloud({
  vault: 'knowledge_base',
  text: 'This is important information about our product.',
  meta: { source: 'documentation', category: 'product' },
  name: 'product_info_v1',
  split: true,
  split_size: 500
});
```

---

### addSite()

```typescript
async addSite(params: AddSiteParams): Promise<any>
```

Scrapes and adds content from a website URL to a vault.

**Parameters:**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `vault` | `string` | `''` | Yes | Target vault name |
| `site` | `string` | `''` | Yes | URL of website to scrape |
| `embeddings_model` | `string` | instance default | No | Embeddings model |

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.addSite({
  vault: 'web_content',
  site: 'https://example.com/blog/article'
});
```

---

### uploadFromJson()

```typescript
async uploadFromJson(vault: string, jsonData: any): Promise<any>
```

Uploads data to a vault from a JSON object.

**Parameters:**
- `vault`: Target vault name
- `jsonData`: JSON object containing data to upload

**Returns:** Promise resolving to response object

**Example:**
```javascript
const data = {
  items: [
    { text: 'Item 1', meta: { id: 1 } },
    { text: 'Item 2', meta: { id: 2 } }
  ]
};

await vv.uploadFromJson('my_vault', data);
```

---

### downloadToJson()

```typescript
async downloadToJson(params: DownloadParams): Promise<any>
```

Downloads vault data as JSON.

**Parameters:**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `vault` | `string` | `''` | Yes | Vault name to download |
| `return_meta` | `boolean` | `false` | No | Include metadata in export |

**Returns:** Promise resolving to JSON data object

**Example:**
```javascript
const data = await vv.downloadToJson({
  vault: 'my_vault',
  return_meta: true
});

console.log(data);
```

---

### getItems()

```typescript
async getItems(vault: string, itemIds: number[]): Promise<any>
```

Retrieves specific items from a vault by their IDs.

**Parameters:**
- `vault`: Vault name
- `itemIds`: Array of item IDs to retrieve

**Returns:** Promise resolving to array of items

**Example:**
```javascript
const items = await vv.getItems('my_vault', [1, 5, 10, 15]);
console.log(items);
```

---

### editItem()

```typescript
async editItem(vault: string, itemId: number, newText: string): Promise<any>
```

Edits the text content of an existing item in a vault.

**Parameters:**
- `vault`: Vault name
- `itemId`: ID of the item to edit
- `newText`: New text content

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.editItem('my_vault', 42, 'Updated text content for this item');
```

---

### deleteItems()

```typescript
async deleteItems(vault: string, itemIds: number[]): Promise<any>
```

Deletes specific items from a vault.

**Parameters:**
- `vault`: Vault name
- `itemIds`: Array of item IDs to delete

**Returns:** Promise resolving to response object

**Example:**
```javascript
// Delete single item
await vv.deleteItems('my_vault', [123]);

// Delete multiple items
await vv.deleteItems('my_vault', [1, 2, 3, 4, 5]);
```

---

### getTotalItems()

```typescript
async getTotalItems(vault: string): Promise<any>
```

Gets the total number of items in a vault.

**Parameters:**
- `vault`: Vault name

**Returns:** Promise resolving to object containing total count

**Example:**
```javascript
const result = await vv.getTotalItems('my_vault');
console.log(`Total items: ${result.total}`);
```

---

## Search & Similarity Methods

### getSimilar()

```typescript
async getSimilar(params: GetSimilarParams): Promise<SimilarityResponse>
```

Finds items semantically similar to the input text. Supports both single-vault and cross-vault search.

**Parameters:**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `text` | `string` | `''` | Yes | Text to find similarities for |
| `vault` | `string` | `''` | No* | Single vault to search (legacy) |
| `vaults` | `string \| string[] \| object` | `null` | No* | Cross-vault search configuration |
| `embeddings_model` | `string` | instance default | No | Embeddings model |
| `num_items` | `number` | `4` | No | Maximum number of results |
| `include_distances` | `boolean` | `false` | No | Include similarity distances |

*Either `vault` or `vaults` should be provided

**Cross-Vault Search (`vaults` parameter):**

1. **Single vault (string):**
   ```javascript
   vaults: 'my_vault'
   ```

2. **Multiple vaults (array):**
   ```javascript
   vaults: ['vault1', 'vault2', 'vault3']
   // Returns top num_items across all vaults, globally sorted
   ```

3. **Multiple vaults with minimums (object):**
   ```javascript
   vaults: { vault1: 3, vault2: 1, vault3: 2 }
   // Guarantees at least N results from each vault
   // Total results may exceed num_items to satisfy minimums
   ```

**Returns:** Promise resolving to similarity results

**Examples:**

```javascript
// Single vault search
const results1 = await vv.getSimilar({
  vault: 'my_vault',
  text: 'machine learning algorithms',
  num_items: 5,
  include_distances: true
});

// Cross-vault search (merged results)
const results2 = await vv.getSimilar({
  vaults: ['vault1', 'vault2'],
  text: 'customer feedback',
  num_items: 10
});

// Cross-vault with minimums per vault
const results3 = await vv.getSimilar({
  vaults: {
    technical_docs: 3,
    user_guides: 2,
    faqs: 1
  },
  text: 'installation process',
  num_items: 10,
  include_distances: true
});

console.log(results3);
```

---

### getDistance()

```typescript
async getDistance(vault: string, id1: number, id2: number): Promise<any>
```

Calculates the semantic distance between two items in a vault.

**Parameters:**
- `vault`: Vault name
- `id1`: First item ID
- `id2`: Second item ID

**Returns:** Promise resolving to object containing distance value

**Example:**
```javascript
const result = await vv.getDistance('my_vault', 10, 25);
console.log(`Distance: ${result.distance}`);
```

---

## Customization Methods

### savePersonalityMessage()

```typescript
async savePersonalityMessage(vault: string, personalityMessage: string): Promise<any>
```

Sets a personality/system message for a vault that influences chat responses.

**Parameters:**
- `vault`: Vault name
- `personalityMessage`: Personality instruction text

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.savePersonalityMessage(
  'customer_support',
  'You are a helpful and empathetic customer support agent. Always be polite and professional.'
);
```

---

### fetchPersonalityMessage()

```typescript
async fetchPersonalityMessage(vault: string): Promise<any>
```

Retrieves the personality message for a vault.

**Parameters:**
- `vault`: Vault name

**Returns:** Promise resolving to object containing personality message

**Example:**
```javascript
const result = await vv.fetchPersonalityMessage('customer_support');
console.log(result.personality_message);
```

---

### saveCustomPrompt()

```typescript
async saveCustomPrompt(vault: string, customPrompt: string, context?: boolean): Promise<any>
```

Saves a custom prompt template for a vault.

**Parameters:**
- `vault`: Vault name
- `customPrompt`: Custom prompt template
- `context` (optional): Whether prompt includes context. Defaults to `true`.

**Returns:** Promise resolving to response object

**Example:**
```javascript
await vv.saveCustomPrompt(
  'my_vault',
  'Answer based on the following context: {context}\n\nQuestion: {question}',
  true
);
```

---

### fetchCustomPrompt()

```typescript
async fetchCustomPrompt(vault: string, context?: boolean): Promise<any>
```

Retrieves the custom prompt for a vault.

**Parameters:**
- `vault`: Vault name
- `context` (optional): Defaults to `true`

**Returns:** Promise resolving to object containing custom prompt

**Example:**
```javascript
const result = await vv.fetchCustomPrompt('my_vault');
console.log(result.custom_prompt);
```

---

## Flow Execution Methods

Flows are complex AI workflows that can include multiple steps, conditional logic, and integrations.

> **📖 For comprehensive information about building flows**, including all available node types, development patterns, and deployment options, see the **VectorFlow Documentation** (`vectorflow_docs.md`). For detailed logging structure of flow executions, see **VectorFlow Logging Reference** (`vectorflow_logging.md`).

### runFlow()

```typescript
async runFlow(
  flowName: string,
  message: string,
  history?: string,
  conversation_user_id?: string | null,
  session_id?: string | null,
  invoke_method?: string | null,
  internal_vars?: Record<string, any> | null,
  callbacks?: FlowCallbacks
): Promise<FlowResult>
```

Executes a flow and returns the complete response when finished.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flowName` | `string` | - | Flow identifier |
| `message` | `string` | - | User message/input |
| `history` | `string` | `''` | Conversation history |
| `conversation_user_id` | `string \| null` | `null` | User ID for multi-user apps |
| `session_id` | `string \| null` | `null` | Session/thread identifier |
| `invoke_method` | `string \| null` | `null` | Label written to logs |
| `internal_vars` | `object \| null` | `null` | Override flow variables |
| `callbacks` | `FlowCallbacks` | `{}` | Callback functions |

**Callbacks Object:**
```typescript
{
  onMessage?: (message: string) => void;
  onLog?: (log: any) => void;
  onError?: (error: any) => void;
}
```

**Returns:** Promise resolving to `FlowResult`:
```typescript
{
  response: string;
  logs: any[];
}
```

**Example:**
```javascript
const result = await vv.runFlow(
  'customer_onboarding',
  'I need help setting up my account',
  '',
  'user_123',
  'session_abc',
  'web_chat',
  { plan_type: 'premium' },
  {
    onMessage: (msg) => console.log('Response:', msg),
    onError: (err) => console.error('Error:', err)
  }
);

console.log('Final response:', result.response);
console.log('Logs:', result.logs);
```

---

### runFlowStream()

```typescript
async runFlowStream(
  flowName: string,
  message: string,
  history?: string,
  conversation_user_id?: string | null,
  session_id?: string | null,
  invoke_method?: string | null,
  internal_vars?: Record<string, any> | null,
  callbacks?: FlowCallbacks
): Promise<FlowResult>
```

Executes a flow with streaming response, calling callbacks as data arrives.

**Parameters:** Same as `runFlow()`

**Returns:** Promise resolving to `FlowResult` after streaming completes

**Example:**
```javascript
let fullResponse = '';

const result = await vv.runFlowStream(
  'tech_support',
  'How do I reset my password?',
  '',
  null,
  null,
  'api',
  { user_tier: 'enterprise' },
  {
    onMessage: (chunk) => {
      fullResponse += chunk;
      process.stdout.write(chunk); // Real-time streaming
    },
    onLog: (log) => {
      console.log('[LOG]', log);
    },
    onError: (error) => {
      console.error('[ERROR]', error);
    }
  }
);

console.log('\n\nFinal result:', result);
```

---

## File Upload Methods

### uploadPdf()

```typescript
async uploadPdf(
  pdfFile: File,
  vault: string,
  options?: PDFUploadOptions
): Promise<any>
```

Uploads and processes a PDF file into a vault.

**Parameters:**
- `pdfFile`: File object (PDF)
- `vault`: Target vault name
- `options` (optional): PDF processing options

**Options Object:**
```typescript
{
  summarize?: boolean;    // Generate summary of PDF
  splitSize?: number;     // Chunk size for splitting
}
```

**Returns:** Promise resolving to response object

**Example (Browser):**
```javascript
// In an HTML file input handler
document.getElementById('pdfInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  
  if (file && file.type === 'application/pdf') {
    const result = await vv.uploadPdf(file, 'documents', {
      summarize: true,
      splitSize: 1000
    });
    
    console.log('PDF uploaded successfully:', result);
  }
});
```

**Example (Node.js with FormData):**
```javascript
import fs from 'fs';
import { Blob } from 'buffer';

// Read PDF file
const pdfBuffer = fs.readFileSync('./document.pdf');
const pdfBlob = new Blob([pdfBuffer], { type: 'application/pdf' });

// Create File object
const pdfFile = new File([pdfBlob], 'document.pdf', { 
  type: 'application/pdf' 
});

await vv.uploadPdf(pdfFile, 'documents', {
  summarize: true,
  splitSize: 500
});
```

---

## Utility Methods

### fetch3DMap()

```typescript
async fetch3DMap(vault: string, highlightId?: number | null): Promise<any>
```

Fetches 3D visualization data for a vault's vector space.

**Parameters:**
- `vault`: Vault name
- `highlightId` (optional): Item ID to highlight in visualization

**Returns:** Promise resolving to 3D map data

**Example:**
```javascript
const mapData = await vv.fetch3DMap('my_vault', 42);
// Use mapData with a 3D visualization library
```

---

### makeAuthenticatedRequest()

```typescript
async makeAuthenticatedRequest(
  url: string,
  options?: RequestInit,
  maxRetries?: number
): Promise<Response>
```

Internal method for making authenticated HTTP requests with automatic retry logic and token refresh. Generally not called directly by users.

**Parameters:**
- `url`: Request URL
- `options` (optional): Fetch API options
- `maxRetries` (optional): Maximum retry attempts. Defaults to `2`.

**Returns:** Promise resolving to Response object

---

## Type Definitions

### FlowCallbacks

```typescript
interface FlowCallbacks {
  onMessage?: (partialMessage: string) => void;
  onLog?: (logData: any) => void;
  onError?: (error: any) => void;
}
```

### FlowResult

```typescript
interface FlowResult {
  response: string;
  logs: any[];
}
```

### PDFUploadOptions

```typescript
interface PDFUploadOptions {
  summarize?: boolean;
  splitSize?: number;
}
```

### GetSimilarParams

```typescript
interface GetSimilarParams {
  embeddings_model?: string | null;
  vault?: string;
  text: string;
  num_items?: number;
  include_distances?: boolean;
  vaults?: string | string[] | Record<string, number> | null;
}
```

### GetChatParams

```typescript
interface GetChatParams {
  vault?: string;
  embeddings_model?: string;
  text?: string;
  history?: string | null;
  summary?: boolean;
  get_context?: boolean;
  n_context?: number;
  return_context?: boolean;
  smart_history_search?: boolean;
  model?: string;
  include_context_meta?: boolean;
  custom_prompt?: string | boolean;
  temperature?: number;
  timeout?: number;
}
```

### AddCloudParams

```typescript
interface AddCloudParams {
  vault: string;
  embeddings_model?: string;
  text: string;
  meta?: any | null;
  name?: string | null;
  split?: boolean;
  split_size?: number;
  gen_sum?: boolean;
}
```

---

## Error Handling

All asynchronous methods return Promises and can throw errors. Always use proper error handling:

### Using async/await (Recommended)

```javascript
try {
  const response = await vv.getChat({
    vault: 'my_vault',
    text: 'Hello'
  });
  console.log(response);
} catch (error) {
  console.error('Error:', error.message);
}
```

### Using .then().catch()

```javascript
vv.getChat({
  vault: 'my_vault',
  text: 'Hello'
})
  .then(response => {
    console.log(response);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });
```

### Common Error Types

| Error | Description | Solution |
|-------|-------------|----------|
| "Login failed" | Invalid credentials | Check email and password |
| "Deployment initialization failed" | Invalid deployment ID | Verify deployment ID |
| "Session expired" | Access token expired and refresh failed | Call `login()` again |
| "Request failed: 404" | Endpoint not found | Check API method and parameters |
| "Request failed: 401" | Unauthorized | Re-authenticate |
| Network errors | Connection issues | Check internet connection, retry |

### Automatic Token Refresh

The client automatically handles token refresh when:
- Token is within 60 seconds of expiration
- Request receives a 401 Unauthorized response

If token refresh fails, you'll need to re-authenticate:

```javascript
try {
  await vv.getVaults();
} catch (error) {
  if (error.message.includes('Session expired')) {
    // Re-authenticate
    await vv.login(email, password);
    // Retry operation
    await vv.getVaults();
  }
}
```

---

## Complete Example

Here's a complete example demonstrating common workflows:

```javascript
import VectorVault from 'vectorvault';

async function main() {
  // Initialize
  const vv = new VectorVault();
  
  try {
    // Authenticate
    await vv.login('user@example.com', 'password123');
    console.log('✓ Logged in successfully');
    
    // Create a vault
    await vv.createVault('product_docs');
    console.log('✓ Vault created');
    
    // Add some data
    await vv.addCloud({
      vault: 'product_docs',
      text: 'Our product uses advanced AI technology to analyze customer feedback.',
      meta: { source: 'docs', version: '1.0' },
      split: true
    });
    console.log('✓ Data added');
    
    // Get similar items
    const similar = await vv.getSimilar({
      vault: 'product_docs',
      text: 'How does the AI analysis work?',
      num_items: 3,
      include_distances: true
    });
    console.log('✓ Similar items:', similar);
    
    // Get a chat response with context
    let fullResponse = '';
    await vv.getChatStream(
      {
        vault: 'product_docs',
        text: 'Explain the AI features',
        get_context: true,
        n_context: 3,
        model: 'gpt-4o'
      },
      (chunk) => {
        fullResponse += chunk;
        process.stdout.write(chunk);
      }
    );
    console.log('\n✓ Chat completed');
    
    // Get vault statistics
    const total = await vv.getTotalItems('product_docs');
    console.log(`✓ Total items in vault: ${total.total}`);
    
    // List all vaults
    const vaults = await vv.getVaults();
    console.log('✓ Available vaults:', vaults);
    
  } catch (error) {
    console.error('Error:', error.message);
  } finally {
    // Logout when done
    vv.logout();
    console.log('✓ Logged out');
  }
}

main();
```

---

## Additional Resources

### Documentation
- **VectorFlow Documentation** (`vectorflow_docs.md`) - Comprehensive guide to building and deploying AI agent flows
- **VectorFlow Logging Reference** (`vectorflow_logging.md`) - Detailed logging structure for all flow node types

### Links
- **GitHub Repository**: https://github.com/John-Rood/vectorvault-js
- **NPM Package**: https://www.npmjs.com/package/vectorvault
- **Official Website**: https://vectorvault.io
- **VectorFlow Builder**: https://app.vectorvault.io/vector-flow
- **Support**: https://github.com/John-Rood/vectorvault-js/issues

---

## License

MIT License - See LICENSE file for details

---

**Last Updated:** November 2025  
**Version:** 1.6.5

