"""Session lifecycle management."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    model_name: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    status: str = "active"
    generation_config: dict = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)

    def touch(self):
        self.last_accessed = time.time()


class SessionManager:
    """Manages active video generation sessions."""

    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def create_session(self, model_name: str, config: Optional[dict] = None) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            model_name=model_name,
            generation_config=config or {},
        )
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for model {model_name}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    def end_session(self, session_id: str) -> Optional[Session]:
        session = self.sessions.pop(session_id, None)
        if session:
            session.status = "ended"
            logger.info(f"Ended session {session_id}")
        return session

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "model_name": s.model_name,
                "status": s.status,
                "created_at": s.created_at,
                "last_accessed": s.last_accessed,
                "output_files": s.output_files,
            }
            for s in self.sessions.values()
        ]

    def get_sessions_for_model(self, model_name: str) -> list[Session]:
        return [s for s in self.sessions.values() if s.model_name == model_name]
