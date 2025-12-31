"""
Conversation Manager
Handles conversation memory and session state
"""
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_complete: bool = True


@dataclass
class ConversationSession:
    """A conversation session"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Manages conversation state and memory
    
    Features:
    - Rolling memory (keeps last N turns)
    - Session tracking
    - Turn-level logging
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        session_timeout_minutes: int = 30
    ):
        """
        Initialize conversation manager
        
        Args:
            max_turns: Maximum turns to keep in memory
            session_timeout_minutes: Session timeout for cleanup
        """
        self.max_turns = max_turns
        self.session_timeout = session_timeout_minutes
        
        self._sessions: Dict[str, ConversationSession] = {}
        self._current_session_id: Optional[str] = None
        
        # Current incomplete turn being built
        self._pending_user_text = ""
        self._pending_assistant_text = ""
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = ConversationSession(session_id=session_id)
        self._sessions[session_id] = session
        self._current_session_id = session_id
        
        logger.info(f"[CONV] Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[ConversationSession]:
        """Get a session by ID or current session"""
        sid = session_id or self._current_session_id
        if sid:
            return self._sessions.get(sid)
        return None
    
    def add_user_turn(self, text: str, session_id: Optional[str] = None):
        """Add a completed user turn"""
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session()
            session = self._sessions[session_id]
        
        turn = ConversationTurn(role="user", content=text)
        session.turns.append(turn)
        session.last_activity = datetime.now()
        
        # Trim to max turns
        if len(session.turns) > self.max_turns * 2:
            session.turns = session.turns[-self.max_turns * 2:]
        
        logger.debug(f"[CONV] User: {text[:50]}...")
    
    def add_assistant_turn(self, text: str, session_id: Optional[str] = None):
        """Add a completed assistant turn"""
        session = self.get_session(session_id)
        if not session:
            return
        
        turn = ConversationTurn(role="assistant", content=text)
        session.turns.append(turn)
        session.last_activity = datetime.now()
        
        # Trim to max turns
        if len(session.turns) > self.max_turns * 2:
            session.turns = session.turns[-self.max_turns * 2:]
        
        logger.debug(f"[CONV] Assistant: {text[:50]}...")
    
    def update_pending_user(self, text: str):
        """Update pending (incomplete) user text"""
        self._pending_user_text = text
    
    def update_pending_assistant(self, text: str):
        """Update pending (incomplete) assistant text"""
        self._pending_assistant_text = text
    
    def commit_pending_user(self):
        """Commit pending user text as complete turn"""
        if self._pending_user_text:
            self.add_user_turn(self._pending_user_text)
            self._pending_user_text = ""
    
    def commit_pending_assistant(self):
        """Commit pending assistant text as complete turn"""
        if self._pending_assistant_text:
            self.add_assistant_turn(self._pending_assistant_text)
            self._pending_assistant_text = ""
    
    def clear_pending(self):
        """Clear all pending text (e.g., on interruption)"""
        self._pending_user_text = ""
        self._pending_assistant_text = ""
    
    def get_recent_turns(
        self,
        n: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation turns
        
        Args:
            n: Number of turns to get (None = all up to max)
            session_id: Session ID
            
        Returns:
            List of {"role": str, "content": str} dicts
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        count = n or self.max_turns
        recent = session.turns[-count:]
        
        return [{"role": t.role, "content": t.content} for t in recent]
    
    def get_context_string(self, session_id: Optional[str] = None) -> str:
        """
        Get conversation context as a string
        
        Useful for including in RAG prompts
        """
        turns = self.get_recent_turns(session_id=session_id)
        if not turns:
            return ""
        
        lines = []
        for turn in turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        
        return "\n".join(lines)
    
    def end_session(self, session_id: Optional[str] = None):
        """End and cleanup a session"""
        sid = session_id or self._current_session_id
        if sid and sid in self._sessions:
            del self._sessions[sid]
            logger.info(f"[CONV] Ended session: {sid}")
            
            if self._current_session_id == sid:
                self._current_session_id = None
    
    def cleanup_old_sessions(self):
        """Remove sessions older than timeout"""
        now = datetime.now()
        to_remove = []
        
        for sid, session in self._sessions.items():
            age_minutes = (now - session.last_activity).total_seconds() / 60
            if age_minutes > self.session_timeout:
                to_remove.append(sid)
        
        for sid in to_remove:
            self.end_session(sid)
            logger.info(f"[CONV] Cleaned up expired session: {sid}")
    
    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id
    
    @property
    def turn_count(self) -> int:
        """Get current session turn count"""
        session = self.get_session()
        return len(session.turns) if session else 0
