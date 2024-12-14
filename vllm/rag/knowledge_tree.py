from typing import Dict, List, Optional, Set, Tuple
import torch
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentKVCache:
    """Stores KV cache for a document along with its hash and metadata"""
    doc_hash: str
    kv_cache: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    token_ids: List[int]
    
class KnowledgeTreeNode:
    def __init__(self):
        self.children: Dict[str, KnowledgeTreeNode] = {}
        self.kv_cache: Optional[DocumentKVCache] = None
        self.prefix_hash: str = ""
        
class KnowledgeTree:
    def __init__(self, max_cache_size: int = 1000):
        self.root = KnowledgeTreeNode()
        self.max_cache_size = max_cache_size
        self.cache_size = 0
        self.doc_hashes: Set[str] = set()
        
    def _compute_doc_hash(self, content: str) -> str:
        """Compute hash of document content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compute_prefix_hash(self, parent_hash: str, doc_hash: str) -> str:
        """Compute hash of prefix + current document"""
        return hashlib.sha256(f"{parent_hash}:{doc_hash}".encode()).hexdigest()
        
    def insert(self, 
              docs: List[str],
              kv_caches: List[torch.Tensor],
              hidden_states: Optional[List[torch.Tensor]] = None,
              token_ids: Optional[List[List[int]]] = None) -> None:
        """Insert documents and their KV caches into the tree"""
        current = self.root
        prefix_hash = ""
        
        for i, (doc, kv_cache) in enumerate(zip(docs, kv_caches)):
            doc_hash = self._compute_doc_hash(doc)
            prefix_hash = self._compute_prefix_hash(prefix_hash, doc_hash)
            
            if doc_hash not in self.doc_hashes:
                self.cache_size += 1
                self.doc_hashes.add(doc_hash)
                
                # Evict oldest cache if size exceeds limit
                if self.cache_size > self.max_cache_size:
                    self._evict_oldest()
            
            if doc_hash not in current.children:
                current.children[doc_hash] = KnowledgeTreeNode()
            
            current = current.children[doc_hash]
            current.prefix_hash = prefix_hash
            current.kv_cache = DocumentKVCache(
                doc_hash=doc_hash,
                kv_cache=kv_cache,
                hidden_states=hidden_states[i] if hidden_states else None,
                token_ids=token_ids[i] if token_ids else []
            )
    
    def lookup(self, docs: List[str]) -> Tuple[Optional[List[DocumentKVCache]], str]:
        """Look up KV caches for a sequence of documents"""
        current = self.root
        prefix_hash = ""
        caches = []
        
        for doc in docs:
            doc_hash = self._compute_doc_hash(doc)
            prefix_hash = self._compute_prefix_hash(prefix_hash, doc_hash)
            
            if doc_hash not in current.children:
                return None, prefix_hash
                
            current = current.children[doc_hash]
            caches.append(current.kv_cache)
            
        return caches, prefix_hash
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry using LRU policy"""
        if self.doc_hashes:
            oldest = next(iter(self.doc_hashes))
            self.doc_hashes.remove(oldest)
            self.cache_size -= 1
