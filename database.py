import sqlite3
import json
import logging
import os
from PIL import Image  # Required for type checking

# === CONFIGURATION ===
DB_DIRECTORY = os.path.abspath("db")  # Absolute path for better reliability
HISTORY_LIMIT = 20  # Number of turns (user + model messages) to keep / retrieve

# === LOGGER SETUP ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def ensure_db_directory_exists():
    """Ensures the database directory exists."""
    if not os.path.exists(DB_DIRECTORY):
        try:
            os.makedirs(DB_DIRECTORY)
            logger.info(f"✅ Database directory created at: {DB_DIRECTORY}")
        except OSError as e:
            logger.error(f"❌ Error creating database directory '{DB_DIRECTORY}': {e}", exc_info=True)
            raise
    else:
        logger.debug(f"📂 Database directory already exists at: {DB_DIRECTORY}")


def get_db_path(chat_id: str) -> str:
    """Constructs the path for a chat-ID specific database file."""
    return os.path.join(DB_DIRECTORY, f"chat_{str(chat_id)}.db")


def _init_db_for_chat_if_needed(db_path: str, chat_id_for_log: str):
    """Initializes the chat_history table in the specified DB file if it doesn't exist."""
    conn = None
    try:
        logger.info(f"🔌 Connecting to DB at {db_path} for chat_id {chat_id_for_log}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'model')),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                parts_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp
            ON chat_history (timestamp DESC)
        """)
        conn.commit()
        logger.info(f"✅ DB initialized for chat_id {chat_id_for_log}")
    except sqlite3.Error as e:
        logger.error(f"❌ Error initializing DB for chat_id {chat_id_for_log}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


def get_db_connection(chat_id: str) -> sqlite3.Connection | None:
    """Establishes a connection to the chat-ID specific SQLite database."""
    str_chat_id = str(chat_id)
    db_path = get_db_path(str_chat_id)

    try:
        _init_db_for_chat_if_needed(db_path, str_chat_id)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logger.debug(f"✅ DB connection successful for chat_id {str_chat_id}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"❌ Failed DB connection for chat_id {str_chat_id}: {e}", exc_info=True)
        return None


def _serialize_parts_for_db(parts):
    """Serializes a list of parts for database storage."""
    serialized_parts = []
    if not isinstance(parts, list):
        logger.warning("🚨 Invalid parts format, expected list.")
        return json.dumps([{"type": "error", "content": "Invalid parts format"}])

    for part in parts:
        if isinstance(part, Image.Image):
            serialized_parts.append({"type": "image_placeholder", "format": part.format or "unknown", "size": part.size})
            logger.debug(f"🖼️ Image placeholder serialized: {part.format}, {part.size}")
        elif isinstance(part, (str, dict, list, int, float, bool)) or part is None:
            serialized_parts.append(part)
        else:
            logger.warning(f"⚠️ Unsupported part type: {type(part)} - {str(part)[:100]}")
            serialized_parts.append(str(part))
    return json.dumps(serialized_parts)


def add_turn_to_history(chat_id: str, role: str, parts: list):
    """Adds a new turn to the chat history for the given chat_id."""
    str_chat_id = str(chat_id)
    logger.info(f"📥 Adding turn for chat_id {str_chat_id}, role: {role}, parts: {len(parts) if parts else 0}")

    if not parts:
        logger.info("⚠️ Skipping add_turn_to_history due to empty parts.")
        return

    parts_json_str = _serialize_parts_for_db(parts)
    logger.debug(f"📦 Serialized parts: {parts_json_str}")

    conn = get_db_connection(str_chat_id)
    if not conn:
        logger.error(f"❌ Could not connect to DB for chat_id {str_chat_id}")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (chat_id, role, parts_json)
            VALUES (?, ?, ?)
        """, (str_chat_id, role, parts_json_str))
        conn.commit()
        logger.info(f"✅ Turn added for chat_id {str_chat_id}, row ID: {cursor.lastrowid}")
    except sqlite3.Error as e:
        logger.error(f"❌ Failed to insert turn: {e}", exc_info=True)
    finally:
        conn.close()


def get_history(chat_id: str, limit: int = HISTORY_LIMIT) -> list:
    """Retrieves the last N turns of chat history for the given chat_id."""
    str_chat_id = str(chat_id)
    history = []
    logger.info(f"📤 Retrieving history for chat_id {str_chat_id}, limit: {limit}")

    conn = get_db_connection(str_chat_id)
    if not conn:
        logger.error("❌ Could not connect to retrieve history.")
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, parts_json
            FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

        for row in reversed(rows):
            try:
                parts = json.loads(row["parts_json"])
                if not isinstance(parts, list):
                    logger.warning("⚠️ Corrupted parts_json, not a list.")
                    parts = [str(parts)]
                history.append({"role": row["role"], "parts": parts})
            except json.JSONDecodeError:
                logger.error(f"❌ JSON decode error for chat_id {str_chat_id}")
                history.append({"role": row["role"], "parts": ["[Error decoding history part]"]})

        logger.info(f"✅ Retrieved {len(history)} turns for chat_id {str_chat_id}")
        return history
    except sqlite3.Error as e:
        logger.error(f"❌ Error retrieving history: {e}", exc_info=True)
        return []
    finally:
        conn.close()


def clear_history(chat_id: str):
    """Deletes all chat history for the given chat_id by deleting the DB file."""
    str_chat_id = str(chat_id)
    db_path = get_db_path(str_chat_id)
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"🗑️ History cleared for chat_id {str_chat_id}, file deleted: {db_path}")
        else:
            logger.info(f"ℹ️ No history file found to delete at: {db_path}")
    except OSError as e:
        logger.error(f"❌ Error deleting history DB file: {e}", exc_info=True)


# === Ensure DB directory exists on import ===
ensure_db_directory_exists()

# === Standalone Test ===
if __name__ == "__main__":
    test_chat_id = "test_user_123"
    print(f"\n🧪 Testing with chat_id: {test_chat_id}")
    print("🔍 Current working directory:", os.getcwd())
    print("📄 Full DB path:", get_db_path(test_chat_id))

    clear_history(test_chat_id)

    print("\n➕ Adding test turns...")
    add_turn_to_history(test_chat_id, "user", [{"text": "Hello from user!"}])
    add_turn_to_history(test_chat_id, "model", [{"text": "Hello from model!"}])
    add_turn_to_history(test_chat_id, "user", [{"type": "image_placeholder", "format": "JPEG", "size": (100, 100)}, {"text": "Here’s a red image"}])

    print("\n📜 Retrieving history...")
    history = get_history(test_chat_id)
    for turn in history:
        print(turn)

    print(f"\n✅ Test complete. DB file should be at: {get_db_path(test_chat_id)}")
