import hashlib
import json
import secrets
from datetime import datetime, timedelta
from typing import Any

from config import AuthConfig, PathsConfig


class AuthManager:
    def __init__(self):
        self.users_file = PathsConfig.DATA_DIR / "users.json"
        self.sessions_file = PathsConfig.DATA_DIR / "sessions.json"
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        PathsConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.users_file.exists():
            self._create_default_user()
        if not self.sessions_file.exists():
            self.sessions_file.write_text("{}")

    def _create_default_user(self):
        default_password = "admin123"
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(default_password, salt)
        users: dict[str, dict[str, Any]] = {
            "admin@intellibank.com": {
                "username": "admin@intellibank.com",
                "password_hash": password_hash,
                "salt": salt,
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        self.users_file.write_text(json.dumps(users, indent=2))
        print(f"Default user created: admin@intellibank.com / {default_password}")

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        return self._hash_password(password, salt) == password_hash

    def load_users(self) -> dict[str, dict[str, Any]]:
        content = self.users_file.read_text()
        return json.loads(content) if content else {}

    def save_users(self, users: dict[str, dict[str, Any]]):
        self.users_file.write_text(json.dumps(users, indent=2))

    def load_sessions(self) -> dict[str, dict[str, Any]]:
        content = self.sessions_file.read_text()
        return json.loads(content) if content else {}

    def save_sessions(self, sessions: dict[str, dict[str, Any]]):
        self.sessions_file.write_text(json.dumps(sessions, indent=2))

    def authenticate(self, username: str, password: str) -> str | None:
        users = self.load_users()
        user = users.get(username)
        if not user:
            return None
        if self._verify_password(password, user["password_hash"], user["salt"]):
            session_token = secrets.token_urlsafe(32)
            sessions = self.load_sessions()
            sessions[session_token] = {
                "username": username,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(minutes=AuthConfig.SESSION_TIMEOUT_MINUTES)).isoformat()
            }
            self.save_sessions(sessions)
            user["last_login"] = datetime.now().isoformat()
            self.save_users(users)
            return session_token
        return None

    def validate_session(self, session_token: str) -> str | None:
        if not session_token:
            return None
        sessions = self.load_sessions()
        session = sessions.get(session_token)
        if not session:
            return None
        if datetime.fromisoformat(session["expires_at"]) < datetime.now():
            del sessions[session_token]
            self.save_sessions(sessions)
            return None
        session["expires_at"] = (datetime.now() + timedelta(minutes=AuthConfig.SESSION_TIMEOUT_MINUTES)).isoformat()
        self.save_sessions(sessions)
        return session.get("username")

    def logout(self, session_token: str):
        sessions = self.load_sessions()
        if session_token in sessions:
            del sessions[session_token]
            self.save_sessions(sessions)

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        users = self.load_users()
        user = users.get(username)
        if not user:
            return False
        if not self._verify_password(old_password, user["password_hash"], user["salt"]):
            return False
        new_salt = secrets.token_hex(16)
        user["password_hash"] = self._hash_password(new_password, new_salt)
        user["salt"] = new_salt
        self.save_users(users)
        return True


auth_manager = AuthManager()
