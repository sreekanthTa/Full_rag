"""Central configuration loader backed by .env.

Reads secrets/variables from the project `.env` file via pydantic-settings.
Every required key is surfaced here so the rest of the app just imports
`settings` instead of touching environment variables directly.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str = ""

    # Mem0 long-term memory. The env file currently stores the key under
    # MEMO_API_KEY (note the O); we map it here in __init__.
    mem0_api_key: str = ""
    mem0_api_url: str = "https://api.mem0.ai"

    # Short-term memory persistence (Postgres). LangGraph's PostgresSaver
    # needs a plain postgresql:// psycopg string; the .env file holds a
    # SQLAlchemy-style postgresql+psycopg:// URL, so we normalize it.
    database_url: str = ""

    # Vector store
    vector_db_path: str = str(BASE_DIR / "chroma_db")
    collection_name: str = "rag_pdf_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def groq_ready(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def mem0_ready(self) -> bool:
        return bool(self.mem0_api_key)

    @property
    def postgres_ready(self) -> bool:
        return bool(self.database_url)

    @property
    def postgres_conn_string(self) -> str:
        """Plain psycopg connection string (postgresql://...), which is what
        PostgresSaver expects. The .env uses SQLAlchemy's
        postgresql+psycopg:// scheme."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _read_env(key: str) -> str:
    """Read the first occurrence of a key from the .env file directly.

    Used for values pydantic could not bind (duplicate keys in the file).
    """
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


settings = Settings()

# The .env has TWO DATABASE_URL entries (Neon then Supabase). pydantic binds
# only the last one it sees from the env file, so we explicitly read the FIRST
# occurrence (Neon, per the user's choice) via _read_env.
settings.database_url = _read_env("DATABASE_URL")

# If MEM0_API_KEY is empty, fall back to the MEMO_API_KEY value that is
# already present in the .env file (the key is Mem0-format "m0-...").
if not settings.mem0_api_key:
    settings.mem0_api_key = _read_env("MEMO_API_KEY")