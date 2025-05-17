import sqlite3
import json
import logging
import os
from PIL import Image  # Required for type checking

# === CONFIGURATION ===
DB_DIRECTORY = os.path.abspath("db")  # Absolute path for better reliability
# Increased HISTORY_LIMIT to remember more turns (e.g., 20 user + 20 model = 40 total parts)
# Adjust this based on typical message length and model token limits.
# Gemini 1.5 Flash has a large context window (1M tokens), so more turns are generally fine.
HISTORY_LIMIT = 40  # Number of turns (user + model messages) to keep / retrieve

# === LOGGER SETUP ===
logger = logging.getLogger(__name__)
# BasicConfig in a module is okay for standalone testing or if it's the main entry point.
# If imported, the root logger configuration from the main application (main.py) will typically apply.
logging.basicConfig(level=logging.INFO) # Changed to INFO for less verbose default logging

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
        logger.debug(f"🔌 Attempting to connect to DB at {db_path} for chat_id {chat_id_for_log} for init.")
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
        logger.debug(f"✅ DB schema initialized/verified for chat_id {chat_id_for_log}")
    except sqlite3.Error as e:
        logger.error(f"❌ Error initializing DB for chat_id {chat_id_for_log}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


def get_db_connection(chat_id: str) -> sqlite3.Connection | None:
    """Establishes a connection to the chat-ID specific SQLite database."""
    str_chat_id = str(chat_id)
    db_path = get_db_path(str_chat_id)

    ensure_db_directory_exists() # Ensure directory exists before trying to connect/create file
    _init_db_for_chat_if_needed(db_path, str_chat_id) # Ensure table exists

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logger.debug(f"✅ DB connection successful for chat_id {str_chat_id} at {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"❌ Failed DB connection for chat_id {str_chat_id} at {db_path}: {e}", exc_info=True)
        return None


def _serialize_parts_for_db(parts):
    """Serializes a list of parts for database storage. Images become placeholders."""
    serialized_parts = []
    if not isinstance(parts, list):
        # Handle case where 'parts' might be a single string or other non-list item
        # Forcing it into a list with a text part, or a structured error.
        if isinstance(parts, str):
             parts = [parts] # Convert single string to list of one string
        else:
            logger.warning(f"🚨 Invalid parts format, expected list, got {type(parts)}. Wrapping as error string.")
            return json.dumps([{"type": "error", "content": f"Invalid parts format: {str(parts)}"}])

    for part in parts:
        if isinstance(part, Image.Image):
            # Store a placeholder for images, not the image data itself in JSON
            serialized_parts.append({
                "type": "image_placeholder",
                "format": part.format or "unknown",
                "size": part.size
            })
            logger.debug(f"🖼️ Image placeholder serialized: {part.format}, {part.size}")
        elif isinstance(part, (str, dict, list, int, float, bool)) or part is None:
            serialized_parts.append(part)
        else:
            # Fallback for any other types not explicitly handled
            logger.warning(f"⚠️ Unsupported part type for DB serialization: {type(part)} - converting to string: {str(part)[:100]}")
            serialized_parts.append({"type": "unsupported", "content": str(part)})
    return json.dumps(serialized_parts)


def add_turn_to_history(chat_id: str, role: str, parts: list):
    """Adds a new turn to the chat history for the given chat_id."""
    str_chat_id = str(chat_id)
    logger.debug(f"📥 Attempting to add turn for chat_id {str_chat_id}, role: {role}, parts count: {len(parts) if parts else 0}")

    if not parts:
        logger.info(f"⚠️ Skipping add_turn_to_history for chat_id {str_chat_id} due to empty parts.")
        return

    parts_json_str = _serialize_parts_for_db(parts)
    logger.debug(f"📦 Serialized parts for DB (chat_id {str_chat_id}): {parts_json_str}")

    conn = get_db_connection(str_chat_id)
    if not conn:
        logger.error(f"❌ Could not connect to DB for chat_id {str_chat_id}. Turn not added.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (chat_id, role, parts_json)
            VALUES (?, ?, ?)
        """, (str_chat_id, role, parts_json_str))
        conn.commit()
        logger.info(f"✅ Turn added for chat_id {str_chat_id} (role: {role}), DB row ID: {cursor.lastrowid}")
    except sqlite3.Error as e:
        logger.error(f"❌ Failed to insert turn into DB for chat_id {str_chat_id}: {e}", exc_info=True)
    finally:
        conn.close()


def get_history(chat_id: str, limit: int = HISTORY_LIMIT) -> list:
    """
    Retrieves the last N turns of chat history for the given chat_id.
    Each turn is a dict: {'role': 'user'/'model', 'parts': [deserialized_parts_list]}
    Image parts in history are placeholders, not actual image data.
    """
    str_chat_id = str(chat_id)
    history = []
    logger.debug(f"📤 Attempting to retrieve history for chat_id {str_chat_id}, limit: {limit} turns.")

    conn = get_db_connection(str_chat_id)
    if not conn:
        logger.error(f"❌ Could not connect to DB to retrieve history for chat_id {str_chat_id}.")
        return []

    try:
        cursor = conn.cursor()
        # Fetch rows, then reverse in Python to maintain chronological order for the model
        cursor.execute("""
            SELECT role, parts_json
            FROM chat_history
            WHERE chat_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (str_chat_id, limit))
        rows = cursor.fetchall()

        # Reverse rows to get chronological order (oldest first) for the model's history
        for row_data in reversed(rows):
            try:
                # Ensure row_data is a dictionary-like object (sqlite3.Row behaves like one)
                role = row_data["role"]
                parts_json = row_data["parts_json"]
                
                deserialized_parts = json.loads(parts_json)
                
                # The 'parts' for the Gemini model should be a list of simple items (text, image objects).
                # Our DB stores JSON representations. We need to ensure 'deserialized_parts'
                # is suitable for the model. For text and placeholders, it is.
                # If images were stored as base64 and needed to be reloaded as PIL Images,
                # this would be the place to do it. But we store placeholders.
                history.append({"role": role, "parts": deserialized_parts})

            except json.JSONDecodeError as je:
                logger.error(f"❌ JSON decode error for a history part in chat_id {str_chat_id}: {je}. Part: {row_data['parts_json'][:100]}")
                history.append({"role": row_data["role"], "parts": ["[Error decoding history part]"]})
            except KeyError as ke:
                logger.error(f"❌ KeyError accessing row data for chat_id {str_chat_id}: {ke}. Row: {row_data}")
                history.append({"role": "error", "parts": ["[Error processing history row data]"]})


        logger.info(f"✅ Retrieved {len(history)} history turns for chat_id {str_chat_id} (limit was {limit}).")
        return history
    except sqlite3.Error as e:
        logger.error(f"❌ Error retrieving history from DB for chat_id {str_chat_id}: {e}", exc_info=True)
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
            logger.info(f"🗑️ History DB file deleted for chat_id {str_chat_id} at: {db_path}")
        else:
            logger.info(f"ℹ️ No history DB file found to delete for chat_id {str_chat_id} at: {db_path}")
    except OSError as e:
        logger.error(f"❌ Error deleting history DB file for chat_id {str_chat_id}: {e}", exc_info=True)


# === Ensure DB directory exists on import/first call ===
# ensure_db_directory_exists() # Called before each connection now for robustness

# === Standalone Test ===
if __name__ == "__main__":
    # Configure logger for standalone test
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    test_chat_id = "test_user_standalone_123"
    print(f"\n🧪 Testing database module with chat_id: {test_chat_id}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"📂 DB Directory: {DB_DIRECTORY}")
    print(f"📄 Full DB path for test: {get_db_path(test_chat_id)}")

    print("\n[Action] Ensuring DB directory exists...")
    ensure_db_directory_exists()

    print(f"\n[Action] Clearing any old history for {test_chat_id}...")
    clear_history(test_chat_id) # Clears by deleting the file

    print("\n[Action] Adding test turns...")
    # Simulating PIL Image object for placeholder serialization
    class MockPILImage:
        def __init__(self, format, size):
            self.format = format
            self.size = size

    mock_image = MockPILImage("JPEG", (100,100))

    add_turn_to_history(test_chat_id, "user", ["Hello from user! First message."])
    add_turn_to_history(test_chat_id, "model", ["Hello back from model!"])
    add_turn_to_history(test_chat_id, "user", [mock_image, "Here’s an image I sent."])
    add_turn_to_history(test_chat_id, "model", ["Nice image! I'll remember it with a placeholder."])
    add_turn_to_history(test_chat_id, "user", ["What was the image about again?"])


    print("\n[Action] Retrieving history (all turns)...")
    history = get_history(test_chat_id, limit=10) # Use a limit higher than added turns
    if history:
        for i, turn in enumerate(history):
            print(f"  Turn {i+1}: Role: {turn['role']}, Parts: {turn['parts']}")
    else:
        print("  No history retrieved.")

    print(f"\n[Action] Retrieving history (limit 2 turns from end)...")
    history_limited = get_history(test_chat_id, limit=2) # Should give last 2 turns (1 user, 1 model)
    if history_limited:
        for i, turn in enumerate(history_limited):
            print(f"  Limited Turn {i+1}: Role: {turn['role']}, Parts: {turn['parts']}")
    else:
        print("  No limited history retrieved.")

    print(f"\n✅ Test complete. DB file should be at: {get_db_path(test_chat_id)}")
    print("Check the 'db' directory and the console output for logs.")