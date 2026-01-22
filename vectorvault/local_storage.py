# VECTOR VAULT CONFIDENTIAL
# __________________
# 
#  All Rights Reserved.
# 
# NOTICE:  All information contained herein is, and remains
# the property of Vector Vault and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Vector Vault
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Vector Vault. See license for consent.

"""
Local filesystem storage backend for VectorVault.
Provides the same interface as CloudManager for seamless swapping.
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, List, Any


class LocalStorageManager:
    """
    Local filesystem storage backend for VectorVault.
    Provides the same interface as CloudManager for seamless swapping.
    """
    
    def __init__(self, vault: str, base_dir: str = None):
        self.vault = vault
        self.base_dir = Path(base_dir or os.path.expanduser('~/.vectorvault'))
        self.vault_dir = self.base_dir / vault
        self._lock = Lock()
        
        # Cached data (mirrors CloudManager attributes)
        self.item_mapping: Dict[str, str] = {}
        self.vault_metadata: Dict[str, Any] = {}
        self.custom_prompt_with_context: Optional[str] = None
        self.custom_prompt_no_context: Optional[str] = None
        self.personality_message: Optional[str] = None
        
        # Ensure directories exist and load data
        self._ensure_dirs()
        self.load_init_data()
    
    # ==================== Directory Management ====================
    
    def _ensure_dirs(self):
        """Create necessary directory structure."""
        dirs = [
            self.vault_dir,
            self.vault_dir / 'items',
            self.vault_dir / 'meta',
            self.vault_dir / 'storage',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _atomic_write(self, filepath: Path, content: str):
        """Write content atomically using temp file + rename."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        temp_path.rename(filepath)
    
    def _safe_read(self, filepath: Path) -> Optional[str]:
        """Safely read file content, returning None if not found."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    # ==================== Init Data ====================
    
    def load_init_data(self):
        """Load all cached data from local files (mirrors CloudManager.load_init_data)."""
        # Load mapping
        mapping_content = self._safe_read(self.vault_dir / 'mapping.json')
        if mapping_content:
            self.item_mapping = json.loads(mapping_content)
        
        # Load vault metadata
        meta_content = self._safe_read(self.vault_dir / 'vault_meta.json')
        if meta_content:
            self.vault_metadata = json.loads(meta_content)
        
        # Load prompts
        prompts_content = self._safe_read(self.vault_dir / 'prompts.json')
        if prompts_content:
            prompts = json.loads(prompts_content)
            self.custom_prompt_with_context = prompts.get('with_context')
            self.custom_prompt_no_context = prompts.get('no_context')
            self.personality_message = prompts.get('personality')
    
    # ==================== Vault Operations ====================
    
    def vault_exists(self, vault_name: str = None) -> bool:
        """Check if vault directory exists or if vectors file exists."""
        if vault_name and vault_name.endswith('.ann'):
            # Check for vectors file specifically
            return (self.vault_dir / 'vectors.ann').exists()
        return self.vault_dir.exists() and (self.vault_dir / 'mapping.json').exists()
    
    def list_vaults(self, prefix: str = '') -> List[str]:
        """List all vaults (directories in base_dir)."""
        vaults = []
        if self.base_dir.exists():
            for p in self.base_dir.iterdir():
                if p.is_dir() and (p / 'mapping.json').exists():
                    if not prefix or p.name.startswith(prefix):
                        vaults.append(p.name)
        return sorted(vaults)
    
    def delete(self):
        """Delete entire vault directory."""
        if self.vault_dir.exists():
            shutil.rmtree(self.vault_dir)
        self.item_mapping = {}
        self.vault_metadata = {}
        
        # Update vaults list
        vaults = self.download_vaults_list_from_cloud()
        if self.vault in vaults:
            vaults.remove(self.vault)
            self.upload_vaults_list(vaults)
    
    # ==================== Item Operations ====================
    
    def upload(self, item_uuid: str, text: str, meta: dict, vault: str = None):
        """Store item text and metadata."""
        target_dir = self.base_dir / (vault or self.vault)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / 'items').mkdir(parents=True, exist_ok=True)
        (target_dir / 'meta').mkdir(parents=True, exist_ok=True)
        
        # Save item text
        item_path = target_dir / 'items' / f'{item_uuid}.txt'
        self._atomic_write(item_path, text)
        
        # Save item metadata
        meta_path = target_dir / 'meta' / f'{item_uuid}.json'
        self._atomic_write(meta_path, json.dumps(meta, indent=2))
    
    def download_text_from_cloud(self, path: str) -> Optional[str]:
        """
        Download text from "cloud" path.
        Translates cloud paths to local paths.
        Path format: vault/uuid/type or similar patterns.
        """
        # Handle various path formats
        parts = path.replace('\\', '/').split('/')
        
        # Try to extract vault, uuid, and type from path
        # Common patterns: "vault/uuid/item", "vault/uuid/meta", or just file paths
        if len(parts) >= 2:
            # Check if this looks like vault/uuid/type format
            vault_name = parts[0]
            item_uuid = parts[1]
            
            target_dir = self.base_dir / vault_name
            
            if len(parts) > 2:
                file_type = parts[2]
                if file_type == 'item':
                    return self._safe_read(target_dir / 'items' / f'{item_uuid}.txt')
                elif file_type == 'meta':
                    return self._safe_read(target_dir / 'meta' / f'{item_uuid}.json')
            
            # If no type specified, try both
            result = self._safe_read(target_dir / 'items' / f'{item_uuid}.txt')
            if result is None:
                result = self._safe_read(target_dir / 'meta' / f'{item_uuid}.json')
            return result
        
        return None
    
    def upload_to_cloud(self, path: str, content: str):
        """Upload content to "cloud" path."""
        parts = path.replace('\\', '/').split('/')
        
        if len(parts) >= 2:
            vault_name = parts[0]
            item_uuid = parts[1]
            
            target_dir = self.base_dir / vault_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if len(parts) > 2:
                file_type = parts[2]
                if file_type == 'item':
                    (target_dir / 'items').mkdir(parents=True, exist_ok=True)
                    filepath = target_dir / 'items' / f'{item_uuid}.txt'
                elif file_type == 'meta':
                    (target_dir / 'meta').mkdir(parents=True, exist_ok=True)
                    filepath = target_dir / 'meta' / f'{item_uuid}.json'
                else:
                    filepath = target_dir / '/'.join(parts[1:])
            else:
                filepath = target_dir / item_uuid
            
            self._atomic_write(filepath, content)
    
    def delete_item(self, item_uuid: str):
        """Delete an item and its metadata."""
        if not item_uuid:
            return
            
        item_path = self.vault_dir / 'items' / f'{item_uuid}.txt'
        meta_path = self.vault_dir / 'meta' / f'{item_uuid}.json'
        
        if item_path.exists():
            item_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    
    def item_exists(self, item_uuid: str) -> bool:
        """Check if item exists."""
        if not item_uuid:
            return False
        return (self.vault_dir / 'items' / f'{item_uuid}.txt').exists()
    
    # ==================== Vector Operations ====================
    
    def upload_temp_file(self, temp_file_path: str, vault_name: str):
        """Copy vectors file from temp to permanent location."""
        # vault_name will be like "vaultname.ann" or similar
        dest = self.vault_dir / 'vectors.ann'
        shutil.copy2(temp_file_path, dest)
    
    def download_to_temp_file(self, vault_name: str) -> str:
        """Copy vectors file to temp location and return path."""
        source = self.vault_dir / 'vectors.ann'
        if not source.exists():
            raise FileNotFoundError(f"Vectors file not found: {source}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ann')
        temp_file.close()
        shutil.copy2(source, temp_file.name)
        return temp_file.name
    
    # ==================== Mapping Operations ====================
    
    def get_mapping(self) -> Dict[str, str]:
        """Get item mapping for current vault."""
        return self.item_mapping.copy()
    
    def save_mapping(self, mapping: Dict[str, str] = None):
        """Save item mapping to file."""
        if mapping is not None:
            self.item_mapping = mapping
        
        filepath = self.vault_dir / 'mapping.json'
        self._atomic_write(filepath, json.dumps(self.item_mapping, indent=2))
    
    # ==================== Vaults List ====================
    
    def download_vaults_list_from_cloud(self) -> List[str]:
        """Get list of all vaults."""
        vaults_file = self.base_dir / 'vaults.json'
        content = self._safe_read(vaults_file)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return self.list_vaults()
    
    def upload_vaults_list(self, vaults: List[str]):
        """Save list of all vaults."""
        vaults_file = self.base_dir / 'vaults.json'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(vaults_file, json.dumps(sorted(set(vaults)), indent=2))
    
    # ==================== Prompts & Personality ====================
    
    def upload_custom_prompt(self, prompt: str, context: bool = True):
        """Save custom prompt."""
        if context:
            self.custom_prompt_with_context = prompt
        else:
            self.custom_prompt_no_context = prompt
        self._save_prompts()
    
    def download_custom_prompt(self, context: bool = True) -> Optional[str]:
        """Get custom prompt."""
        if context:
            return self.custom_prompt_with_context
        return self.custom_prompt_no_context
    
    def upload_personality_message(self, message: str):
        """Save personality message."""
        self.personality_message = message
        self._save_prompts()
    
    def download_personality_message(self) -> Optional[str]:
        """Get personality message."""
        return self.personality_message
    
    def _save_prompts(self):
        """Save all prompts to file."""
        prompts = {
            'with_context': self.custom_prompt_with_context,
            'no_context': self.custom_prompt_no_context,
            'personality': self.personality_message
        }
        filepath = self.vault_dir / 'prompts.json'
        self._atomic_write(filepath, json.dumps(prompts, indent=2))
    
    # ==================== Metadata ====================
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get vault metadata."""
        return self.vault_metadata.copy()
    
    def update_metadata(self, **kwargs):
        """Update vault metadata."""
        self.vault_metadata.update(kwargs)
        filepath = self.vault_dir / 'vault_meta.json'
        self._atomic_write(filepath, json.dumps(self.vault_metadata, indent=2))
    
    def build_update(self, n: int):
        """Update usage stats (mirrors CloudManager.build_update)."""
        self.vault_metadata['last_use'] = time.time()
        self.vault_metadata['total_use'] = self.vault_metadata.get('total_use', 0) + 1
        self.update_metadata()
    
    def build_data_update(self):
        """Update last_update timestamp."""
        self.vault_metadata['last_update'] = time.time()
        self.update_metadata()
    
    # ==================== Cloud API Stubs/Equivalents ====================
    
    @property
    def cloud_api(self):
        """Return self for API compatibility - local mode handles API calls internally."""
        return LocalCloudAPIAdapter(self)
    
    def cloud_name(self, vault: str, item_uuid: str, user: str = None, api: str = None, 
                   item: bool = False, meta: bool = False) -> str:
        """Generate path for item (mirrors cloud_name function)."""
        base = f"{vault}/{item_uuid}"
        if item:
            return f"{base}/item"
        elif meta:
            return f"{base}/meta"
        return base
    
    def download_old_gcs_mapping(self, vault: str) -> Optional[Dict]:
        """Stub for GCS mapping download - not applicable in local mode."""
        return None


class LocalCloudAPIAdapter:
    """
    Adapter that provides CloudAPI-like interface for local storage operations.
    This allows vault.py code to call self.cloud_manager.cloud_api.* methods
    without modification in local mode.
    """
    
    def __init__(self, storage_manager: LocalStorageManager):
        self._storage = storage_manager
    
    def update_vault_mapping(self, vault: str, mapping: Dict):
        """Update vault mapping."""
        target_dir = self._storage.base_dir / vault
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / 'mapping.json'
        self._storage._atomic_write(filepath, json.dumps(mapping, indent=2))
        
        if vault == self._storage.vault:
            self._storage.item_mapping = mapping
    
    def get_vault_mapping(self, vault: str) -> Optional[Dict]:
        """Get vault mapping."""
        if vault == self._storage.vault:
            return self._storage.item_mapping.copy()
        
        target_dir = self._storage.base_dir / vault
        content = self._storage._safe_read(target_dir / 'mapping.json')
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None
    
    def update_vault_metadata(self, vault: str, **kwargs):
        """Update vault metadata."""
        target_dir = self._storage.base_dir / vault
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        meta_file = target_dir / 'vault_meta.json'
        content = self._storage._safe_read(meta_file)
        if content:
            try:
                metadata = json.loads(content)
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = {}
        
        metadata.update(kwargs)
        self._storage._atomic_write(meta_file, json.dumps(metadata, indent=2))
        
        if vault == self._storage.vault:
            self._storage.vault_metadata = metadata
    
    def delete_vault_metadata(self, vault: str):
        """Delete vault metadata."""
        target_dir = self._storage.base_dir / vault
        meta_file = target_dir / 'vault_meta.json'
        if meta_file.exists():
            meta_file.unlink()
    
    def get_user_vault_data(self) -> List[Dict]:
        """Get all vaults metadata."""
        vaults_data = []
        for vault_name in self._storage.list_vaults():
            vault_dir = self._storage.base_dir / vault_name
            meta_file = vault_dir / 'vault_meta.json'
            content = self._storage._safe_read(meta_file)
            if content:
                try:
                    meta = json.loads(content)
                    meta['vault'] = vault_name
                    vaults_data.append(meta)
                except json.JSONDecodeError:
                    vaults_data.append({'vault': vault_name, 'total_items': 0})
            else:
                # Create basic entry from mapping
                mapping_content = self._storage._safe_read(vault_dir / 'mapping.json')
                total_items = 0
                if mapping_content:
                    try:
                        mapping = json.loads(mapping_content)
                        total_items = len(mapping)
                    except json.JSONDecodeError:
                        pass
                vaults_data.append({
                    'vault': vault_name,
                    'total_items': total_items,
                    'last_update': time.time()
                })
        return vaults_data
    
    def update_user_vault_data(self, vault_data: List[Dict]):
        """Update all vaults metadata."""
        for entry in vault_data:
            vault_name = entry.get('vault')
            if vault_name:
                self.update_vault_metadata(
                    vault_name,
                    total_items=entry.get('total_items', 0),
                    last_update=entry.get('last_update', time.time()),
                    last_use=entry.get('last_use', time.time()),
                    total_use=entry.get('total_use', 0)
                )
    
    def save_custom_prompt(self, vault: str, prompt: str, context: bool = True):
        """Save custom prompt for a vault."""
        target_dir = self._storage.base_dir / vault
        target_dir.mkdir(parents=True, exist_ok=True)
        
        prompts_file = target_dir / 'prompts.json'
        content = self._storage._safe_read(prompts_file)
        if content:
            try:
                prompts = json.loads(content)
            except json.JSONDecodeError:
                prompts = {}
        else:
            prompts = {}
        
        if context:
            prompts['with_context'] = prompt
        else:
            prompts['no_context'] = prompt
        
        self._storage._atomic_write(prompts_file, json.dumps(prompts, indent=2))
    
    def save_personality_message(self, vault: str, message: str):
        """Save personality message for a vault."""
        target_dir = self._storage.base_dir / vault
        target_dir.mkdir(parents=True, exist_ok=True)
        
        prompts_file = target_dir / 'prompts.json'
        content = self._storage._safe_read(prompts_file)
        if content:
            try:
                prompts = json.loads(content)
            except json.JSONDecodeError:
                prompts = {}
        else:
            prompts = {}
        
        prompts['personality'] = message
        self._storage._atomic_write(prompts_file, json.dumps(prompts, indent=2))
    
    def run_flow(self, **kwargs):
        """Flows are not supported in local mode."""
        raise NotImplementedError(
            "Flow operations require VectorVault cloud. "
            "Use cloud mode or call LLM directly with get_chat()."
        )
    
    def run_flow_stream(self, **kwargs):
        """Flows are not supported in local mode."""
        raise NotImplementedError(
            "Flow operations require VectorVault cloud. "
            "Use cloud mode or call LLM directly with get_chat_stream()."
        )


class LocalVaultStorageManager:
    """
    Local equivalent of VaultStorageManager for key-value storage.
    """
    
    def __init__(self, vault: str, base_dir: Path):
        self.vault = vault
        self.base_dir = base_dir
        self.storage_dir = base_dir / vault / 'storage'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _atomic_write(self, filepath: Path, content: str):
        """Write content atomically using temp file + rename."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        temp_path.rename(filepath)
    
    def create_directory(self, path: str):
        """Create a directory marker."""
        dir_path = self.storage_dir / path
        dir_path.mkdir(parents=True, exist_ok=True)
        marker = dir_path / '.directory'
        marker.touch()
    
    def create_item(self, path: str, value: str):
        """Create an item with the given value."""
        filepath = self.storage_dir / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(filepath, value)
    
    def list_labels(self, path: str = None) -> List[Dict[str, str]]:
        """List items and directories under path."""
        base = self.storage_dir / path if path else self.storage_dir
        if not base.exists():
            return []
        
        result = []
        for item in base.iterdir():
            if item.name == '.directory':
                continue
            item_type = 'directory' if item.is_dir() else 'item'
            result.append({'name': item.name, 'type': item_type})
        
        return sorted(result, key=lambda x: x['name'])
    
    def get_item(self, path: str) -> str:
        """Get item content."""
        filepath = self.storage_dir / path
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def update_item(self, path: str, new_value: str):
        """Update item content."""
        self.create_item(path, new_value)
    
    def delete_label(self, path: str):
        """Delete item or directory."""
        target = self.storage_dir / path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
