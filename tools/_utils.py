"""
Shared Utilities for Tools
==========================

Common helper functions used across all tools.
"""

from typing import Dict, Any, Optional


def find_document(doc_id: str, documents: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a document by ID or filename.
    
    Args:
        doc_id: Document ID (e.g., 'doc_1') or filename
        documents: Dictionary of documents from agent state
        
    Returns:
        Document info dict or None if not found
    """
    # Direct lookup by ID
    if doc_id in documents:
        return documents[doc_id]
    
    # Fallback: search by filename
    for key, info in documents.items():
        if info.get("filename") == doc_id:
            return info
    
    return None


def get_available_doc_ids(documents: Dict[str, Any]) -> list:
    """Get list of available document IDs for error messages."""
    return list(documents.keys())
