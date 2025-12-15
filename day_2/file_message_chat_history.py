import json
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a local JSON file.

    This implementation persists a list of message dicts and restores them
    back into LangChain message objects using `messages_to_dict` / `messages_from_dict`.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    @property
    def messages(self) -> list:
        """Retrieve all messages from the JSON file as LangChain message objects."""
        if not self.file_path.exists():
            return []
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)  # expects a JSON array of message dicts
            # Convert list[dict] → list[BaseMessage]
            return messages_from_dict(data)
        except (json.JSONDecodeError, OSError):
            # If file is corrupt or unreadable, return empty history
            return []

    def add_messages(self, messages: list) -> None:
        """Append multiple messages to the JSON file.

        `messages` should be a list of LangChain `BaseMessage` objects.
        """
        existing = []
        if self.file_path.exists():
            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        # Convert incoming messages to serializable dicts
        new_dicts = messages_to_dict(messages)
        all_dicts = existing + new_dicts

        # Write back the full list as JSON
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(all_dicts, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """Clear all messages by writing an empty JSON array."""
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump([], f)