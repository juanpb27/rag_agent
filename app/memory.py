from typing import Dict, List

# Global dictionary to store messages by session_id in memory
_memory_store: Dict[str, List[Dict[str, str]]] = {}


def load_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Loads chat history for a specific session.
    """
    return _memory_store.get(session_id, [])


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Saves a message to the history of a specific session.
    """
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    
    message = {
        "role": role,
        "content": content
    }
    
    _memory_store[session_id].append(message)


def clear_memory(session_id: str) -> None:
    """
    Clears all chat history for a specific session.
    """
    if session_id in _memory_store:
        del _memory_store[session_id]
