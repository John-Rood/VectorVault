# ACT Node Security & Allowed Modules

## Overview

The ACT node in VectorFlow provides a secure, sandboxed environment for executing Python code. This document outlines which Python modules are allowed and which are restricted for security reasons.

## Security Model

The ACT node uses a **whitelist approach** - only explicitly approved modules can be imported. This prevents:
- File system access
- System command execution  
- Network exploits
- Code injection attacks
- Resource exhaustion

## ✅ Allowed Modules

### Core Data Processing
```python
import math          # Mathematical functions
import datetime      # Date and time handling
import json          # JSON encoding/decoding
import random        # Random number generation
import re            # Regular expressions
import collections   # Specialized container datatypes
import itertools     # Functions for efficient looping
import functools     # Higher-order functions and operations on callable objects
import operator      # Standard operators as functions
import string        # String constants and classes
import decimal       # Decimal fixed point and floating point arithmetic
import time          # Time-related functions
```

### Text/String Processing
```python
import textwrap      # Text wrapping and formatting
import unicodedata   # Unicode character database
import difflib       # Helpers for comparing sequences
```

### Data Structures & Algorithms
```python
import heapq         # Heap queue algorithm
import bisect        # Array bisection algorithm
import copy          # Shallow and deep copy operations
```

### Encoding/Decoding
```python
import base64        # Base16, Base32, Base64, Base85 data encodings
import binascii      # Convert between binary and ASCII
import html          # HTML utilities
import urllib.parse  # Parse URLs into components
```

### Hashing & Identification
```python
import hashlib       # Secure hash and message digest algorithms
import hmac          # Keyed-Hashing for Message Authentication
import uuid          # UUID objects according to RFC 4122
```

### Data Formats
```python
import csv           # CSV file reading and writing
import xml.etree.ElementTree  # XML processing
```

### System Info (Safe Subset)
```python
import platform      # Access to underlying platform data
```

### Path Operations
```python
import pathlib       # Object-oriented filesystem paths
import mimetypes     # Map filenames to MIME types
```

### Networking (Safe)
```python
import requests      # HTTP library (already approved)
import urllib.request # Extensible library for opening URLs
import urllib.error   # Exception classes raised by urllib.request
```

## ❌ Prohibited Modules

### File System Access
```python
# These modules are BLOCKED for security
import os            # Operating system interface
import shutil        # High-level file operations
import tempfile      # Generate temporary files and directories
import glob          # Unix shell-style pathname pattern expansion
import fnmatch       # Unix filename pattern matching
```

### System/Process Control
```python
# These modules are BLOCKED for security
import sys           # System-specific parameters and functions
import subprocess    # Subprocess management
import signal        # Set handlers for asynchronous events
import resource      # Resource usage information
import atexit        # Exit handlers
```

### Code Execution
```python
# These modules are BLOCKED for security
import pickle        # Python object serialization (can execute arbitrary code)
import marshal       # Internal Python object serialization
import imp           # Access the import internals
import importlib     # The implementation of import
import runpy         # Locating and running Python modules
import code          # Interpreter base classes
import codeop        # Compile Python code
```

### Threading/Process Management
```python
# These modules are BLOCKED to avoid conflicts
import threading     # Thread-based parallelism
import multiprocessing  # Process-based parallelism
import concurrent    # High-level interface for asynchronously executing callables
import asyncio       # Asynchronous I/O
import queue         # A synchronized queue class
```

### Raw Networking
```python
# These modules are BLOCKED for security
import socket        # Low-level networking interface
import ssl           # TLS/SSL wrapper for socket objects
import ftplib        # FTP protocol client
import telnetlib     # Telnet client
import smtplib       # SMTP protocol client
```

### Database Direct Access
```python
# These modules are BLOCKED (use API calls instead)
import sqlite3       # DB-API 2.0 interface for SQLite databases
import dbm           # Interfaces to Unix "databases"
```

## Available Built-in Functions

### ✅ Allowed Built-ins
```python
# Basic functions
print(), len(), range(), enumerate(), zip(), map(), filter()
sorted(), reversed(), min(), max(), sum(), abs(), round()

# Type functions  
int(), float(), str(), bool(), list(), dict(), tuple(), set()

# Safe attribute access
getattr(), setattr(), delattr(), property()

# Global scope access
globals()  # Returns controlled global scope, not system globals
```

### ❌ Blocked Built-ins
```python
# These built-ins are BLOCKED for security
open()      # File operations
input()     # User input (could hang execution)
exec()      # Execute arbitrary code
eval()      # Evaluate arbitrary expressions
compile()   # Compile source into code object
dir()       # List object attributes (information disclosure)
vars()      # Return __dict__ attribute (information disclosure)
locals()    # Return local namespace (information disclosure)
help()      # Interactive help system
exit()      # Exit functions
quit()      # Exit functions
```

## Available Variables

### ✅ Available in Script Environment
```python
MESSAGE    # Current user message
HISTORY    # Conversation history  
save       # Dictionary for saving variables (use save['var_name'] = value)

# All VectorFlow variables set in previous nodes
{your_variable_name}  # Any variables you've set in variable or other nodes
```

## Security Best Practices

### ✅ Safe Patterns
```python
# Data processing
import json
data = json.loads(MESSAGE)
processed = [item.upper() for item in data['items']]
save['result'] = processed

# Time operations
import time
import datetime
save['timestamp'] = datetime.datetime.now().isoformat()
time.sleep(1)  # Safe delays

# Text processing
import re
import textwrap
cleaned_text = re.sub(r'[^\w\s]', '', MESSAGE)
save['formatted'] = textwrap.fill(cleaned_text, width=80)

# HTTP requests
import requests
response = requests.get('https://api.example.com/data', timeout=10)
save['api_data'] = response.json()
```

### ❌ Blocked Patterns
```python
# These will raise ImportError
import os
os.system('rm -rf /')  # BLOCKED

import subprocess  
subprocess.run(['cat', '/etc/passwd'])  # BLOCKED

import pickle
pickle.loads(untrusted_data)  # BLOCKED - can execute arbitrary code

# File operations
with open('/etc/passwd', 'r') as f:  # BLOCKED
    data = f.read()
```

## Error Handling

When blocked modules are imported, you'll see:
```
ImportError: Module 'module_name' is not allowed.
```

Common solutions:
1. **Use allowed alternatives** (e.g., `urllib.request` instead of `socket`)
2. **Use VectorFlow variables** instead of file operations
3. **Use API calls** instead of direct database access
4. **Use the `requests` module** for HTTP operations

## Need Additional Modules?

If you need a module that's not on the allowed list:

1. **Check if there's an allowed alternative** that meets your needs
2. **Consider if the functionality can be achieved** with VectorFlow nodes
3. **Contact your VectorFlow administrator** to request module approval

New modules are evaluated based on:
- Security risk assessment
- Compatibility with VectorFlow architecture  
- Common use case requirements
- Potential for misuse

---

*This security model ensures ACT nodes remain powerful yet safe for all users while maintaining system integrity.* 